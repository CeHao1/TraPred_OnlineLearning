import argparse
import inspect
import json
import math
import multiprocessing
import os
import numpy as np
import pickle
import random
import subprocess
import sys
import time
import pdb
from collections import defaultdict
from multiprocessing import Process
from random import randint
from typing import Dict, List, Tuple, NamedTuple, Any, Union, Optional


def add_argument(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    # Required parameters

    # ============== train, eval mode ============
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run evaluation.")


    # =============== file loading and saving ========
    parser.add_argument("--data_dir",
                        default='train/data/',
                        type=str)
    parser.add_argument("--data_dir_for_val",
                        default='val/data/',
                        type=str)

    parser.add_argument("--output_dir", default="tmp/", type=str)
    parser.add_argument("--log_dir", default=None, type=str)
    parser.add_argument("--temp_file_dir", default=None, type=str)
    parser.add_argument("--temp_file_name", default=None, type=str)
    parser.add_argument("--reuse_temp_file",
                        action='store_true')

    # ============= dataset parameters ===========
    parser.add_argument("--hidden_size",
                        default=64,
                        type=int,
                        help='size of input matrix padding')
    parser.add_argument("--batch_size",
                        default=64,
                        type=int)


    # ============= device parameters ============
    parser.add_argument("--core_num",
                        default=1,
                        type=int)
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--cuda_visible_device_num",
                        default=None,
                        type=int)
    parser.add_argument("--use_map",
                        action='store_true')
    
    # =========== parameters for dataset generation ===========
    parser.add_argument("--max_distance",
                        default=50.0,
                        type=float)
    parser.add_argument("--use_centerline",
                        action='store_true')
    parser.add_argument("--no_agents",
                        action='store_true')


    parser.add_argument("--other_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-ep", "--eval_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("-tp", "--train_params",
                        nargs='*',
                        default=[],
                        type=str)
    parser.add_argument("--not_use_api",
                        action='store_true')
    
    parser.add_argument("--visualize",
                        action='store_true')
    
    parser.add_argument("--add_prefix",
                        default=None)
    parser.add_argument("--multi",
                        nargs='?',
                        default=None,
                        const=6,
                        type=int)
    parser.add_argument("--nms_threshold",
                        default=None,
                        type=float)
    parser.add_argument("--master_port", default='12355')
    parser.add_argument("--argoverse",
                        action='store_true')
    parser.add_argument("--future_frame_num",
                        default=80,
                        type=int)
    parser.add_argument("--mode_num",
                        default=6,
                        type=int)
    parser.add_argument("--dump_data",
                        default=1,
                        type=int)


class Args:
    data_dir = None
    data_dir_for_val = None
    data_kind = None
    seed = None
    batch_size = None
    distributed_training = None
    cuda_visible_device_num = None
    log_dir = None
    learning_rate = None
    do_eval = None
    hidden_size = None
    initializer_range = None
    temp_file_dir = None
    output_dir = None
    use_map = None
    reuse_temp_file = None
    model_recover_path = None
    do_train = None
    max_distance = None
    other_params: Dict = None
    eval_params = None
    train_params = None
    no_agents = None
    not_use_api = None
    core_num = None
    visualize = None
    hidden_dropout_prob = None
    use_centerline = None
    add_prefix = None
    do_test = None
    multi = None
    argoverse = None
    future_frame_num = None
    no_cuda = None
    mode_num = None
    nms_threshold = None

    image_name = None

args: Args = None

logger = None


def init(args_: Args):
    global args
    args = args_

    if args.do_eval:
        assert os.path.exists(args.output_dir)
        assert os.path.exists(args.data_dir_for_val)
    else:
        assert os.path.exists(args.data_dir)

    if args.log_dir is None:
        args.log_dir = args.output_dir
    if args.temp_file_dir is None:
        args.temp_file_dir = './temp_file'

    dic = {}
    for i, param in enumerate(args.other_params + args.eval_params + args.train_params):
        if '=' in param:
            index = str(param).index('=')
            key = param[:index]
            value = param[index + 1:]
            # key, value = param.split('=')
            dic[key] = value if not str(value).isdigit() else int(value)
        else:
            dic[param] = True
    args.other_params = dic

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.temp_file_dir, exist_ok=True)
    if not args.do_eval:
        src_dir = os.path.join(args.output_dir, 'src')
        if os.path.exists(src_dir):
            subprocess.check_output('rm -r {}'.format(src_dir), shell=True, encoding='utf-8')
        os.makedirs(src_dir, exist_ok=False)
        for each in os.listdir('src'):
            is_dir = '-r' if os.path.isdir(os.path.join('src', each)) else ''
            subprocess.check_output(f'cp {is_dir} {os.path.join("src", each)} {src_dir}', shell=True, encoding='utf-8')
        with open(os.path.join(src_dir, 'cmd'), 'w') as file:
            file.write(' '.join(sys.argv))
    args.model_save_dir = os.path.join(args.output_dir, 'model_save')
    os.makedirs(args.model_save_dir, exist_ok=True)

    if args.do_eval:
        args.data_dir = args.data_dir_for_val
        args.do_train = False

    print(dict(sorted(vars(args_).items())))
    args_dict = vars(args)
    print()
    for each in ['output_dir', 'other_params']:
        if each in args_dict:
            temp = args_dict[each]
            if each == 'other_params':
                temp = [param if args.other_params[param] is True else (param, args.other_params[param]) for param in
                        args.other_params]
            print("\033[31m" + each + "\033[0m", temp)
    logging(vars(args_), type='args', is_json=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.join(args.temp_file_dir, time_begin), exist_ok=True)

    assert args.do_train or args.do_eval


def add_eval_param(param):
    if param not in args.eval_params:
        args.eval_params.append(param)


def get_name(name='', append_time=False):
    if name.endswith(time_begin):
        return name
    
    if args.do_train:
        prefix = 'train'
    elif args.do_eval:
        prefix = 'eval'
    prefix = args.add_prefix + '.' + prefix if args.add_prefix is not None else prefix
    suffix = '.' + time_begin if append_time else ''
    return prefix + str(name) + suffix


eps = 1e-5

origin_point = None
origin_angle = None


def get_pad_vector(li):
    """
    Pad vector to length of args.hidden_size
    """
    assert len(li) <= args.hidden_size
    li.extend([0] * (args.hidden_size - len(li)))
    return li


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


def batch_list_to_batch_tensors_old(batch):
    batch_tensors = []
    for x in zip(*batch):
        batch_tensors.append(x)
    return batch_tensors


def round_value(v):
    return round(v / 100)


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))

def get_dists(arr):
    dis= np.sqrt(arr[:,0]**2 + arr[:,1]**2)
    return dis


def get_dis_point2point(point, point_=(0.0, 0.0)):
    return np.sqrt(np.square((point[0] - point_[0])) + np.square((point[1] - point_[1])))


def get_angle(x, y):
    return math.atan2(y, x)

# not use
def get_sub_matrix(traj, object_type, x=0, y=0, angle=None):
    res = []
    for i in range(0, len(traj), 2):
        if i > 0:
            vector = [traj[i - 2] - x, traj[i - 1] - y, traj[i] - x, traj[i + 1] - y]
            if angle is not None:
                vector[0], vector[1] = rotate(vector[0], vector[1], angle)
                vector[2], vector[3] = rotate(vector[2], vector[3], angle)
            res.append(vector)
    return res


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


def rotate_(x, y, cos, sin):
    res_x = x * cos - y * sin
    res_y = x * sin + y * cos
    return res_x, res_y

def get_unit_vector(point_a, point_b):
    der_x = point_b[0] - point_a[0]
    der_y = point_b[1] - point_a[1]
    scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
    der_x *= scale
    der_y *= scale
    return (der_x, der_y)


idx_in_batch_2_ans_points = {}
idx_in_batch_2_ans_point_scores = {}

def get_subdivide_points(polygon, include_self=False, threshold=1.0, include_beside=False, return_unit_vectors=False):
    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    average_dis = 0
    for i, point in enumerate(polygon):
        if i > 0:
            average_dis += get_dis(point, point_pre)
        point_pre = point
    average_dis /= len(polygon) - 1

    points = []
    if return_unit_vectors:
        assert not include_self and not include_beside
        unit_vectors = []
    divide_num = 1
    while average_dis / divide_num > threshold:
        divide_num += 1
    for i, point in enumerate(polygon):
        if i > 0:
            for k in range(1, divide_num):
                def get_kth_point(point_a, point_b, ratio):
                    return (point_a[0] * (1 - ratio) + point_b[0] * ratio,
                            point_a[1] * (1 - ratio) + point_b[1] * ratio)

                points.append(get_kth_point(point_pre, point, k / divide_num))
                if return_unit_vectors:
                    unit_vectors.append(get_unit_vector(point_pre, point))
        if include_self or include_beside:
            points.append(point)
        point_pre = point
    if include_beside:
        points_ = []
        for i, point in enumerate(points):
            if i > 0:
                der_x = point[0] - point_pre[0]
                der_y = point[1] - point_pre[1]
                scale = 1 / math.sqrt(der_x ** 2 + der_y ** 2)
                der_x *= scale
                der_y *= scale
                der_x, der_y = rotate(der_x, der_y, math.pi / 2)
                for k in range(-2, 3):
                    if k != 0:
                        points_.append((point[0] + k * der_x, point[1] + k * der_y))
                        if i == 1:
                            points_.append((point_pre[0] + k * der_x, point_pre[1] + k * der_y))
            point_pre = point
        points.extend(points_)
    if return_unit_vectors:
        return points, unit_vectors
    return points

index_file = 0

file2pred = {}


def __iter__(self):  # iterator to load data
    for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
        batch = []
        for __ in range(self.batch_size):
            idx = randint(0, len(self.ex_list) - 1)
            batch.append(self.__getitem__(idx))
        # To Tensor
        yield batch_list_to_batch_tensors(batch)


files_written = {}


def logging(*inputs, prob=1.0, type='1', is_json=False, affi=True, sep=' ', to_screen=False, append_time=False, as_pickle=False):
    """
    Print args into log file in a convenient style.
    """
    if to_screen:
        print(*inputs, sep=sep)
    if not random.random() <= prob or not hasattr(args, 'log_dir'):
        return

    file = os.path.join(args.log_dir, get_name(type, append_time))
    if as_pickle:
        with open(file, 'wb') as pickle_file:
            assert len(inputs) == 1
            pickle.dump(*inputs, pickle_file)
        return
    if file not in files_written:
        with open(file, "w", encoding='utf-8') as fout:
            files_written[file] = 1
    inputs = list(inputs)
    the_tensor = None
    for i, each in enumerate(inputs):
            the_tensor = each
    np.set_printoptions(threshold=np.inf)

    with open(file, "a", encoding='utf-8') as fout:
        if is_json:
            for each in inputs:
                print(json.dumps(each, indent=4), file=fout)
        elif affi:
            print(*tuple(inputs), file=fout, sep=sep)
            if the_tensor is not None:
                print(json.dumps(the_tensor.tolist()), file=fout)
            print(file=fout)
        else:
            print(*tuple(inputs), file=fout, sep=sep)
            print(file=fout)

def larger(a, b):
    return a > b + eps


def equal(a, b):
    return True if abs(a - b) < eps else False

def rot(verts, rad):
    rad = -rad
    verts = np.array(verts)
    rotMat = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    transVerts = verts.dot(rotMat)
    return transVerts

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def get_start_time_index_from_file_name(file_name):
    start_time = os.path.split(file_name)[1].split('.')[0].split('_')[1]
    return start_time

def get_case_name_from_file_name(file_name):
    case_name = os.path.split(file_name)[1].split('.')[0].split('_')[0]
    return case_name

def get_name_and_time_from_file_name(file_name):
    return get_case_name_from_file_name(file_name), get_start_time_index_from_file_name(file_name)

def get_time():
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

def assert_(satisfied, info=None):
    if not satisfied:
        if info is not None:
            print(info)
        print(sys._getframe().f_code.co_filename, sys._getframe().f_back.f_lineno)
    assert satisfied


time_begin = get_time()


def get_color_text(text, color='red'):
    if color == 'red':
        return "\033[31m" + text + "\033[0m"
    else:
        assert False