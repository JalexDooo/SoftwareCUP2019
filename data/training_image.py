import os
import cv2
import sys
import glob
import numpy  as np
import pandas as pd

from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T

class digitOCR(data.Dataset):

	def __init__(self, root, transforms=None, train=True, test=False):

		self.root = root
		self.transforms = transforms
		self.train = train
		self.test = test

		

		self.data_path = glob.glob(self.root+'/*')
		self.data_path = sorted(self.data_path)

		if self.train:
			self.data_path = self.data_path[:int(0.7*len(self.data_path))]
		elif self.test:
			self.data_path = self.data_path[int(0.7*len(self.data_path)):]


		# for data in self.data_path:


		# print(self.data_path)


	def __getitem__(self, index):

		image_path = 

		return index

	def __len__(self):

		return len(self.data)
		






if __name__ == '__main__':

	training_path = './images'

	test = digitOCR(root=training_path)



