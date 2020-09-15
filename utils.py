import numpy as np
import os

class average_meter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def val(self):
        return self.avg

def get_mnist_index(target, one_class_label):
    train_indices = []
    test_indices = []

    for i in range(len(target)):
        if target[i] == one_class_label:
            train_indices.append(i)
        else:
            test_indices.append(i)

    return train_indices, test_indices

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("Directory " , dir_name ,  " Created ")
    else:    
        pass

