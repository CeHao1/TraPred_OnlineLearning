import src.utils as utils
from src.utils import get_name, get_angle, logging, rotate, round_value, get_pad_vector, get_dis
from src.utils import larger, equal, get_subdivide_points, assert_
from src.utils import get_case_name_from_file_name
import math
import numpy as np
from argoverse.map_representation.map_api import ArgoverseMap

from src.general_utils import AttrDict, ParamDict
import copy
import os

TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3


def generate_one_case(data_dir, args, argoverse_map, ego_id='AV', agent_id='AGENT', start_time=0, input_length=10, predict_length=20,
                     offline_traj=None):
    predict_start = start_time + input_length
    predict_end = start_time + input_length + predict_length  

    hp = ParamDict({
        'start_time': start_time,
        'input_length':input_length,
        'predict_length':predict_length,
        'predict_start':predict_start,
        'predict_end':predict_end,
        'offline_start': 20,
    })

    data = load_data(data_dir)
    return argoverse_get_instance(data, data_dir, args, hp, argoverse_map, offline_traj)

def load_data(file):
    if file.endswith("csv"):
        with open(file, "r", encoding='utf-8') as fin:
            lines = fin.readlines()[1:]
    return lines

def argoverse_get_instance(lines, file_name, args, hp, argoverse_map, offline_traj):
    """
    Extract polylines from one example file content.
    """

    global am
    am = argoverse_map

    global max_vector_num
    vector_num = 0
    id2info = {}
    mapping = {}

    # improve the file name with start time
    b = os.path.split(file_name)
    start_time_string = '{:0>2d}'.format(hp.start_time)
    c = b[1].split('.')[0] + '_' + start_time_string + '.' +b[1].split('.')[1]
    d = os.path.join(b[0], c)
    new_file_name = d

    mapping['file_name'] = new_file_name

    time_step_count = 0
    last_time = -1
    init_title = True

    for i, line in enumerate(lines):

        line = line.strip().split(',')

        if float(line[TIMESTAMP]) > last_time:
            time_step_count += 1
            last_time = float(line[TIMESTAMP])

        if time_step_count <= hp.start_time or time_step_count > hp.predict_end:
            continue


        if init_title:
            mapping['start_time'] = float(line[TIMESTAMP])
            mapping['city_name'] = line[CITY_NAME]
            init_title = False

        line[TIMESTAMP] = float(line[TIMESTAMP]) - mapping['start_time']
        line[X] = float(line[X])
        line[Y] = float(line[Y])
        id = line[TRACK_ID]

        if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
            line[TRACK_ID] = line[OBJECT_TYPE]

        if line[TRACK_ID] in id2info:
            id2info[line[TRACK_ID]].append(line)
            vector_num += 1
        else:
            id2info[line[TRACK_ID]] = [line]

        if line[OBJECT_TYPE] == 'AGENT' and len(id2info['AGENT']) == hp.input_length:
            assert 'AV' in id2info
            assert 'cent_x' not in mapping
            agent_lines = id2info['AGENT']
            mapping['cent_x'] = agent_lines[-1][X]
            mapping['cent_y'] = agent_lines[-1][Y]
            mapping['agent_pred_index'] = len(agent_lines)
            mapping['two_seconds'] = line[TIMESTAMP]
            if 'direction' in args.other_params:
                span = agent_lines[-6:]
                intervals = [2]
                angles = []
                for interval in intervals:
                    for j in range(len(span)):
                        if j + interval < len(span):
                            der_x, der_y = span[j + interval][X] - span[j][X], span[j + interval][Y] - span[j][Y]
                            angles.append([der_x, der_y])

            der_x, der_y = agent_lines[-1][X] - agent_lines[-2][X], agent_lines[-1][Y] - agent_lines[-2][Y]

    if vector_num > max_vector_num:
        max_vector_num = vector_num

    if 'cent_x' not in mapping:
        return None

    origin_labels = np.zeros([hp.predict_length, 2])
    for i, line in enumerate(id2info['AGENT'][hp.input_length:hp.input_length + hp.predict_length]):
        origin_labels[i][0], origin_labels[i][1] = line[X], line[Y]
    mapping['origin_labels'] = origin_labels

    angle = -get_angle(der_x, der_y) + math.radians(90)
    if 'direction' in args.other_params:
        angles = np.array(angles)
        der_x, der_y = np.mean(angles, axis=0)
        angle = -get_angle(der_x, der_y) + math.radians(90)

    mapping['angle'] = angle
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[X], line[Y] = rotate(line[X] - mapping['cent_x'], line[Y] - mapping['cent_y'], angle)
        if 'scale' in mapping:
            scale = mapping['scale']
            line[X] *= scale
            line[Y] *= scale

    return preprocess(args, id2info, mapping, hp)


def preprocess(args, id2info, mapping, hp):
    """
    This function calculates matrix based on information from get_instance.
    """

    polyline_spans = []
    keys = list(id2info.keys())
    assert 'AV' in keys
    assert 'AGENT' in keys
    keys.remove('AV')
    keys.remove('AGENT')
    keys = ['AGENT', 'AV'] + keys
    vectors = []
    two_seconds = mapping['two_seconds']
    mapping['trajs'] = []
    mapping['agents'] = []

    for id in keys:
        polyline = {}

        info = id2info[id]
        start = len(vectors)
        if args.no_agents:
            if id != 'AV' and id != 'AGENT':
                break

        agent = []
        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            agent.append((line[X], line[Y]))

        if args.visualize:
            traj = np.zeros([args.hidden_size])
            for i, line in enumerate(info):
                if larger(line[TIMESTAMP], two_seconds):
                    traj = traj[:i * 2].copy()
                    break
                traj[i * 2], traj[i * 2 + 1] = line[X], line[Y]
                if i == len(info) - 1:
                    traj = traj[:(i + 1) * 2].copy()
            traj = traj.reshape((-1, 2))
            mapping['trajs'].append(traj)

        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            x, y = line[X], line[Y]
            if i > 0:
                # print(x-line_pre[X], y-line_pre[Y])
                vector = [line_pre[X], line_pre[Y], x, y, line[TIMESTAMP], line[OBJECT_TYPE] == 'AV',
                          line[OBJECT_TYPE] == 'AGENT', line[OBJECT_TYPE] == 'OTHERS', len(polyline_spans), i]
                vectors.append(get_pad_vector(vector))
            line_pre = line

        end = len(vectors)
        if end - start == 0:
            assert id != 'AV' and id != 'AGENT'
        else:
            mapping['agents'].append(np.array(agent))

            polyline_spans.append([start, end])

    assert_(len(mapping['agents']) == len(polyline_spans))

    assert len(vectors) <= max_vector_num

    t = len(vectors)
    mapping['map_start_polyline_idx'] = len(polyline_spans)
    if args.use_map:
        vectors, polyline_spans = get_sub_map(args, mapping['cent_x'], mapping['cent_y'], mapping['city_name'],
                                              vectors=vectors,
                                              polyline_spans=polyline_spans, mapping=mapping)


    matrix = np.array(vectors)

    labels = []
    info = id2info['AGENT']
    info = info[mapping['agent_pred_index']:]
    assert len(info) == hp.predict_length
    labels = [0.0 for _ in range(hp.predict_length* 2)]

    if 'set_predict' in args.other_params:
        if 'test' in args.data_dir[0]:
            labels = [0.0 for _ in range(60)]

    if 'goals_2D' in args.other_params:
        point_label = np.array(labels[-2:])
        mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))

        if 'stage_one' in args.other_params:
            stage_one_label = 0
            polygons = mapping['polygons']
            min_dis = 10000.0
            for i, polygon in enumerate(polygons):
                temp = np.min(get_dis(polygon, point_label))
                if temp < min_dis:
                    min_dis = temp
                    stage_one_label = i

            mapping['stage_one_label'] = stage_one_label

    # return mapping

    mapping.update(dict(
        matrix=matrix,
        labels=np.array(labels).reshape([hp.predict_length, 2]),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=hp.predict_length,
    ))

    return mapping

def get_sub_map(args: utils.Args, x, y, city_name, vectors=[], polyline_spans=[], mapping=None):
    """
    Calculate lanes which are close to (x, y) on map.

    Only take lanes which are no more than args.max_distance away from (x, y).

    """

    if args.not_use_api:
        pass
    else:
        assert isinstance(am, ArgoverseMap)
        if 'semantic_lane' in args.other_params:
            lane_ids = am.get_lane_ids_in_xy_bbox(x, y, city_name, query_search_range_manhattan=args.max_distance)
            # lane_centerline = am.city_lane_centerlines_dict[city_name][lane_ids[0]].centerline
            # print(lane_centerline.shape, am.get_ground_height_at_xy(lane_centerline, city_name))
            local_lane_centerlines = [am.get_lane_segment_centerline(lane_id, city_name) for lane_id in lane_ids]
            polygons = local_lane_centerlines
            # z = am.get_ground_height_at_xy(np.array([[x, y]]), city_name)[0]

            if args.visualize:
                angle = mapping['angle']
                vis_lanes = [am.get_lane_segment_polygon(lane_id, city_name)[:, :2] for lane_id in lane_ids]
                t = []
                for each in vis_lanes:
                    for point in each:
                        point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                    num = len(each) // 2
                    t.append(each[:num].copy())
                    t.append(each[num:num * 2].copy())
                vis_lanes = t
                mapping['vis_lanes'] = vis_lanes
        else:
            polygons = am.find_local_lane_centerlines(x, y, city_name,
                                                      query_search_range_manhattan=args.max_distance)
        polygons = [polygon[:, :2].copy() for polygon in polygons]
        angle = mapping['angle']
        for index_polygon, polygon in enumerate(polygons):
            for i, point in enumerate(polygon):
                point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                if 'scale' in mapping:
                    assert 'enhance_rep_4' in args.other_params
                    scale = mapping['scale']
                    point[0] *= scale
                    point[1] *= scale

        if args.use_centerline:
            if 'semantic_lane' in args.other_params:
                local_lane_centerlines = [polygon for polygon in polygons]

        def dis_2(point):
            return point[0] * point[0] + point[1] * point[1]

        def get_dis(point_a, point_b):
            return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

        def get_dis_for_points(point, polygon):
            dis = np.min(np.square(polygon[:, 0] - point[0]) + np.square(polygon[:, 1] - point[1]))
            return np.sqrt(dis)

        def ok_dis_between_points(points, points_, limit):
            dis = np.inf
            for point in points:
                dis = np.fmin(dis, get_dis_for_points(point, points_))
                if dis < limit:
                    return True
            return False

        def get_hash(point):
            return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

        lane_idx_2_polygon_idx = {}
        for polygon_idx, lane_idx in enumerate(lane_ids):
            lane_idx_2_polygon_idx[lane_idx] = polygon_idx

        if 'goals_2D' in args.other_params:
            points = []
            visit = {}
            point_idx_2_unit_vector = []

            mapping['polygons'] = polygons

            for index_polygon, polygon in enumerate(polygons):
                for i, point in enumerate(polygon):
                    hash = get_hash(point)
                    if hash not in visit:
                        visit[hash] = True
                        points.append(point)

                if 'subdivide' in args.other_params:
                    subdivide_points = get_subdivide_points(polygon)
                    points.extend(subdivide_points)
                    subdivide_points = get_subdivide_points(polygon, include_self=True)

            mapping['goals_2D'] = np.array(points)

        for index_polygon, polygon in enumerate(polygons):
            assert_(2 <= len(polygon) <= 10, info=len(polygon))
            # assert len(polygon) % 2 == 1

            # if args.visualize:
            #     traj = np.zeros((len(polygon), 2))
            #     for i, point in enumerate(polygon):
            #         traj[i, 0], traj[i, 1] = point[0], point[1]
            #     mapping['trajs'].append(traj)

            start = len(vectors)
            if 'semantic_lane' in args.other_params:
                assert len(lane_ids) == len(polygons)
                lane_id = lane_ids[index_polygon]
                lane_segment = am.city_lane_centerlines_dict[city_name][lane_id]
            assert_(len(polygon) >= 2)
            for i, point in enumerate(polygon):
                if i > 0:
                    vector = [0] * args.hidden_size
                    vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                    vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                    vector[-5] = 1
                    vector[-6] = i

                    vector[-7] = len(polyline_spans)

                    if 'semantic_lane' in args.other_params:
                        vector[-8] = 1 if lane_segment.has_traffic_control else -1
                        vector[-9] = 1 if lane_segment.turn_direction == 'RIGHT' else \
                            -1 if lane_segment.turn_direction == 'LEFT' else 0
                        vector[-10] = 1 if lane_segment.is_intersection else -1
                    point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                    if i >= 2:
                        point_pre_pre = polygon[i - 2]
                    vector[-17] = point_pre_pre[0]
                    vector[-18] = point_pre_pre[1]

                    vectors.append(vector)
                point_pre = point

            end = len(vectors)
            if start < end:
                polyline_spans.append([start, end])

    return (vectors, polyline_spans)