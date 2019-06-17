import numpy as np
import random
import math


import torch

class GenDataIter():
    
    def __init__(self, data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0
        self.reset()
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)
    
    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        #get all the data in a batch
        index = self.indices[self.idx:self.idx+self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.tensor(d)
        
        #0 is prended to d as a start symbol
        data = torch.cat([torch.zeros(len(index), 1, dtype=torch.int64), d], dim=1)
        target = torch.cat([d, torch.zeros(len(index), 1, dtype=torch.int64)], dim=1)
        
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            #convert string into integer
            l = [int(s) for s in list(line.strip().split())]
            lis.append(l)
        return lis

class DisDataIter():
    
    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_list = self.read_file(real_data_file)
        fake_data_list = self.read_file(fake_data_file)
        self.data = real_data_list + fake_data_list
        self.labels = [1 for _ in range(len(real_data_list))] +\
                        [0 for _ in range(len(fake_data_list))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs) 
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num)/self.batch_size))
        self.idx = 0
        self.reset()

    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        return self

    def __next__(self):
        return self.next() 

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)
    
    def next(self):
        if self.idx >= self.data_num:
            raise StopIteration
        #get all the data in a batch
        index = self.indices[self.idx:self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.tensor(data)
        label = torch.tensor(label)
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            #convert string into integer
            l = [int(s) for s in list(line.strip().split())]
            lis.append(l)
        return lis 