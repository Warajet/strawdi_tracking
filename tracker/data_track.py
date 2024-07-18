import configparser
import csv
import os
import os.path as osp
from PIL import Image
import numpy as np


import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from tracker.data_obj_detect import sort_files


class StrawDISequence(Dataset):
    """
      A class used to represent the video sequence used for Multiple Object Tracking

      - This class will be passed into DataLoader used for testing the multiple strawberry
      tracking implemented by us.


      Attributes
      ----------
      root : str
          - Path to the corresponding StrawDI dataset
      transforms : torchvision.transforms
          - Series of transforms object used to transform or augment data for training 
          or inference of strawberry detection (default None)
      sequences: list(str)
          - List of directory to each of video sequence
      img_paths: list(str)
          - Sorted List of absolute path to each of image sample in the video (dimension: 756 x 1008 x 3)

      Methods
      -------
      __getitem__(idx):
          - Get the pair of an image at the specific index
      
      __len__():
          - Get the number of samples in the Dataset object 
    """
    def __init__(self, root_dir, transform=None):
        """
        Parameters
        ----------
        root : str
          - Path to the corresponding StrawDI dataset
        transforms : torchvision.transforms, optional
          - Series of transforms object used to transform or augment data for training 
        or inference of strawberry detection (default None)
        """
        self.root_dir = root_dir
        self.transform = transform

        # Get list of sequences
        self.sequences = sorted(os.listdir(root_dir))
        self._image_paths = sort_files([os.path.join(root_dir, image_path) for image_path in os.listdir(root_dir)])


    def __len__(self):
        """
        Get the number of samples in the Dataset object
        
        Returns
        -------
        int:
            - number of samples in the Dataset
        
        """
        return len(self._image_paths)

    def __getitem__(self, idx):
        """
        Get the pair of an image at the specific index

        Parameters
        ----------
        idx : int
            The index of the image and groundtruth label in the list
        
        Returns
        -------
        img: list(torch.Tensor([3, H, W]))
            - the image tensor at the specified index in the image path list
            - H: Height of the image (756)
            - W: Width of the image (1008)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self._image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image