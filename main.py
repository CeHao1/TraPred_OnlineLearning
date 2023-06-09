import argparse
import os
from argoverse.map_representation.map_api import ArgoverseMap

import src.utils as utils
from src.scenario import ScenarioArgoverse
from src.general_utils import ParamDict

import torch 
from src.dataset_pytorch import Dataset_Pytorch
import torch.distributed as dist
from torch.utils.data import SequentialSampler, DistributedSampler

import mindspore
from src.dataset_mindspore import Dataset_MindSpore
import mindspore.dataset as ds

# ================================ pytorch implementation ===============================
def get_dataloader_pytorch(args, files, sampler, hp = ParamDict()):
    '''
    The use can change the hp to adjust the parameters in the data generator
    '''
    scenario = ScenarioArgoverse(files, hp)
    dataset = scenario.generate_dataset(args, dataset_type=Dataset_Pytorch, external_am=am)
    new_sampler = sampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=new_sampler)
    return data_loader

def init(args, rank=0, world_size=1):
    def setup(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.master_port

        # initialize the process group
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    setup(rank, world_size)

def main_pytorch():
    '''
    Example code of using Pytorch dataset
    '''
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args)

    global am
    am = ArgoverseMap()

    files = os.listdir(args.data_dir)
    files = [os.path.join(args.data_dir, file) for file in files]

    # training set, load all files (--do_train)
    if args.do_train:
        init(args)
        train_data_loader = get_dataloader_pytorch(args, files, DistributedSampler)
        return train_data_loader

    # evaluation set, load file one by one (--do_eval)
    if args.do_eval:
        eval_loader_list = []
        for file in files:
            eval_file = [file]
            print('process ', file)
            eval_data_loader = get_dataloader_pytorch(args, eval_file, SequentialSampler)
            eval_loader_list.append(eval_data_loader)
        return eval_loader_list

# ================================ mindspore implementation ===============================
def get_dataloader_mindspore(args, files, sampler, hp = ParamDict()):
    scenario = ScenarioArgoverse(files, hp)
    dataset = scenario.generate_dataset(args, dataset_type=Dataset_MindSpore, external_am=am)
    data_loader = ds.GeneratorDataset(source=dataset, column_names=['cases'], sampler=sampler)
    return data_loader
 
def main_mindspore():
    '''
    Example code of using mindspore dataset
    '''
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    args: utils.Args = parser.parse_args()
    utils.init(args)

    global am
    am = ArgoverseMap()

    files = os.listdir(args.data_dir)
    files = [os.path.join(args.data_dir, file) for file in files]

    # training set, load all files (--do_train)
    if args.do_train:
        sampler = ds.DistributedSampler(num_shards = int(len(files)/args.batch_size), shard_id=0)
        train_data_loader = get_dataloader_mindspore(args, files, sampler)
        return train_data_loader

    # evaluation set, load file one by one (--do_eval)
    if args.do_eval:
        eval_loader_list = []
        for file in files:
            eval_file = [file]
            print('process ', file)
            sampler = ds.SequentialSampler(start_index=0, num_samples=1)
            eval_data_loader = get_dataloader_mindspore(args, eval_file, sampler)
            eval_loader_list.append(eval_data_loader)
        return eval_loader_list


if __name__ == "__main__":
    data_loader = main_pytorch()
    # data_loader = main_mindspore()
