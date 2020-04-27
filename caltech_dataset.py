from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        self.dataset = {}
        self.counter = 0
        classes_dict = {}
        class_counter = 0
        indexes = set(np.loadtxt('Caltech101/'+split+'.txt',dtype=str))
        self.classes = os.listdir(root)
        self.classes.remove('BACKGROUND_Google')
        
        for class_ in self.classes:
            classes_dict[class_] = class_counter
            class_counter += 1
            images = os.listdir(root+'/'+class_)
            for image in images:
                if class_+'/'+image in indexes:
                    self.dataset[self.counter] = (pil_loader(root+'/'+class_+'/'+image),classes_dict[class_])
                    self.counter += 1

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.dataset[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        return self.counter

    def __getsplit__(self, train_size = 0.5):
        images, labels = [], []
        sss = StratifiedShuffleSplit(1,train_size=train_size)

        for item in self.dataset.values():
            images.append(item[0])
            labels.append(item[1])

        for x, y in sss.split(images,labels):
            train_indexes = x
            val_indexes = y 

        return train_indexes, val_indexes
