import os
import pickle
import os
import zlib
import torch

class Dataset_Pytorch(torch.utils.data.Dataset):
    def __init__(self, temp_file_name, temp_file_dir, reuse_temp_file=False):
        '''
        This function helps to save and load dataset as pickle

        temp_file_name: the file of saved dataset
        temp_file_dir: the directory of saved dataset
        resue_temp_file: do not save new dataset, but use the previously saved dataset
        '''

        self.save_dir = os.path.join(temp_file_dir, temp_file_name)

        if not reuse_temp_file:
            self.data_list = []
        else:
            self._read_data()

        super().__init__()

    def add_to_data(self, data_list):
        self.data_list += data_list

    def dump_data(self):
        pickle_file = open(self.save_dir, 'wb')
        pickle.dump(self.data_list, pickle_file)
        pickle_file.close()
        print('data temp file saved to', self.save_dir)

    def _read_data(self):
        pickle_file = open(self.save_dir, 'rb')
        self.data_list = pickle.load(pickle_file)

    def __getitem__(self, idx):
        data_compress = self.data_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return instance

    def __len__(self):
        return len(self.data_list)

