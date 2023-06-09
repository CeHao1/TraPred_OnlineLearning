
from multiprocessing import Process
import multiprocessing
from tqdm import tqdm
import pickle
import os
import zlib

from logging import raiseExceptions
from src.general_utils import ParamDict
from src.case_instance import generate_one_case



class ScenarioArgoverse:

    def __init__(self, files, hp : ParamDict):
        self._files = files
        self._hp = self._get_default_hp()
        self._hp.overwrite(hp)
        
        
    def _get_default_hp(self):
        hp  = ParamDict({
            'start_time'            : 0,
            'input_time_length'     : 10,
            'predict_time_length'   : 20,
            'sliding_window_length' : 5,
            'sample_times'          : 5,
            'resue'                 : False,
            'predicted_traj_dir'    : None,
        })
        return hp

    def generate_dataset(self, args, dataset_type, external_am=None):
        '''
        Iterate cases in the directory and convert them as data.
        '''

        dataset = dataset_type(args.temp_file_name, args.temp_file_dir, args.reuse_temp_file)

        if args.reuse_temp_file:
            return dataset

        files = self._files

        # reusable ArgoverseMap
        am = external_am

        # multiprocessing to load files
        queue_in = multiprocessing.Queue(args.core_num) # input queue
        queue_out = multiprocessing.Queue() # output queue
        case_list = [] # output list

        processes = [Process(target=generate_cases, 
                    args=(queue_in, queue_out, args, self._hp, am)) for _ in range(args.core_num)]

        for each in processes:
            each.start()

        # iterate each file
        print('start processing dataset')
        pbar_input = tqdm(total=len(files))
        for index, file in enumerate(files):
            assert file is not None
            queue_in.put(file) # input a single file to the queue
            pbar_input.update(1)

            # block until the input queue is empty
            while not queue_in.empty():
                pass
        pbar_input.close()
        

        # fetch cases from queue 
        print('start output dataset')
        pbar_output = tqdm(total=len(files) * self._hp.sample_times)
        for i in range(len(files) * self._hp.sample_times ):
            case = queue_out.get()
            if case is not None:
                case_list.append(case)

            pbar_output.update(1)
        pbar_output.close()

        # add new cases to the dataset
        dataset.add_to_data(case_list)

        # add None at the end to input queue to stop the function
        for i in range(args.core_num):
            queue_in.put(None)

        # join each process
        for each in processes:
            each.join(1)

        # save and dump dataset
        if args.dump_data:
            dataset.dump_data()

        return dataset


def generate_cases(queue_in, queue_out, args, hp, am):
    '''
    Generate cases in one file. Each case has different start time.
    And the single case is generated in the function generate_one_case.
    Here we use argoverse dateset and DenseTNT function. 
    '''

    while True:
        # fetch file from input queue
        file = queue_in.get()
        if file is None:
            break

        for index in range(hp.sample_times):
            start_time_idx = hp.start_time + index * hp.sliding_window_length
            instance = generate_one_case(file, args, am, 
                                            ego_id='AV', agent_id='AGENT', start_time=start_time_idx, 
                                            input_length=hp.input_time_length, predict_length=hp.predict_time_length)

            data_compress = zlib.compress(pickle.dumps(instance))
            if instance is not None:
                queue_out.put(data_compress)
            else:
                queue_out.put(None)


