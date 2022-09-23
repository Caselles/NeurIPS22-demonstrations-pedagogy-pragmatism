import threading
import numpy as np
from language.build_dataset import sentence_from_configuration
from utils import language_to_id
import torch
from mpi4py import MPI


"""
the replay buffer here is basically from the openai baselines code

"""


class MultiDemoBuffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size 
        self.current_size = 0

        # create the buffer to store info
        self.buffer = {'demo': [[] for _ in range(self.size)],
        #self.buffer = {'demo': np.empty([self.size, self.env_params['max_demo_size'], 59]),
                       'g': np.empty([self.size, self.env_params['nb_possible_goals']])
                       }


        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_batch(self, batch):
        batch_size = len(batch)
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(batch):
                # store the informations
                self.buffer['demo'][idxs[i]] = e[0]
                self.buffer['g'][idxs[i]] = e[1]



    # sample the data from the replay buffer
    def sample(self, batch_size):

        temp_buffer = {}
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][:self.current_size]

        permutation = torch.randperm(self.current_size)
        indices = permutation[:batch_size].numpy()
        batch = np.array([[temp_buffer['demo'][ind], temp_buffer['g'][ind]] for ind in indices])

        return batch

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx


    # get current buffer without any zeroes
    def get_current_buffer(self, max_size=200):

        current_buffer = {}

        print(self.current_size)
        if not self.current_size < max_size:
            index = np.random.randint(0, self.current_size-(max_size/2))
        for key in self.buffer.keys():
            if self.current_size < max_size:
                current_buffer[key] = self.buffer[key][:self.current_size]
            else:
                current_buffer[key] = self.buffer[key][index:int(index+(max_size/2))]

        
        return current_buffer



    # gather all buffers from all processes to root node
    def gather_demo_encoder_buffers(self):

        demo_encoder_current_buffer = self.get_current_buffer()
        data = MPI.COMM_WORLD.gather(demo_encoder_current_buffer, root=0) 

        return data

    # prepare data for training in root node
    def prepare_for_training(self, data):

        self.buffer = {'demo':[], 'g':[]}

        for buf in data:
            for i in range(len(buf['demo'])):
                self.buffer['demo'].append(buf['demo'][i])
                self.buffer['g'].append(buf['g'][i])

        self.current_size = len(self.buffer['demo'])


        return True