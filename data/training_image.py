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
		else:
			self.data_path = self.data_path[int(0.7*len(self.data_path)):]

		for path in self.data_path:
			image = cv2.imread(path, cv2.IMREAD_COLOR)
			if image.shape[0] != 46:
				print('111')
			if image.shape[1] != 120:
				print('222')
			if image.shape[2] != 3:
				print('333')
			# print(image.shape)

		if transforms is None:
			self.transforms = T.Compose([
					T.ToTensor()
				])



	def __getitem__(self, index): # 46 * 120 * 3 -> 4 * 46 * 30 * 3

		image_path = self.data_path[index]

		data = cv2.imread(image_path, cv2.IMREAD_COLOR)

		data = self.transforms(data)

		label = image_path.split('/')[-1][:4]



		return data, label

	def __len__(self):

		return len(self.data)
		






if __name__ == '__main__':

	training_path = './images'

	test = digitOCR(root=training_path)

	print(test[0])
	image, label = test[0]

	image = (image.numpy()*255).astype('uint8')
	image = np.transpose(image, (1, 2, 0))
	print(image)
	cv2.imshow('test', image)
	cv2.waitKey()



