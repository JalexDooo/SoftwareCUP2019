import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import convolve2d
from scipy.stats import gaussian_kde as kde

train_root = './test_images'

# images = glob.glob(train_root+'/*')
# print(images)

class OCR_dataset:

	def __init__(self):
		self.train_images = glob.glob(train_root+'/*')
		# print(self.train_images)
		for i in self.train_images:
			image = mpimg.imread(i)
			image = self.normalize(image)

			image = self.conv2d(image)
			# print(image)
			# plt.hist(image.ravel(), bins=256, range=(0.0, 255.0), fc='k', ec='k')
			
			plt.imshow(image, cmap='gray')
			plt.show()


	def normalize(self, image):
		gray = np.array([0.299, 0.587, 0.114])
		image = np.dot(image, gray)
		# image **= 2
		image = ((image - image.min())/(image.max() - image.min())*255).round()
		return image


	def conv2d(self, image):
		sobel_x = np.c_[
			[-1,-1,1],
			[-1,0,1],
			[-1,1,1]
		]

		sobel_y = np.c_[
			[1,2,1],
			[0,0,0],
			[-1,-2,-1]
		]

		image = convolve2d(image, sobel_x, mode="same", boundary='symm')

		return image


	def test(self):
		image = mpimg.imread(self.train_images[0])
		gray = np.array([0.229, 0.587, 0.114])
		image = np.dot(image, gray)
		image = image**2
		image = ((image.max() - image)/(image.max() - image.min())*255).round()
		
		d0 = kde(image.reshape(-1), bw_method=0.2)(range(256))
		d = np.diff(d0)
		d1 = np.where((d[:-1]<0) * (d[1:]>0))[0]




		# gradient_image = np.gradient(image)
		# plt.hist(image.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')


		sobel_x = np.c_[
			[-1,0,1],
			[-2,0,-2],
			[-1,0,1]
		]

		image = convolve2d(image, sobel_x, mode="same", boundary='symm')


		plt.imshow(image, cmap='gray')
		plt.show()


if __name__ == '__main__':
	dataset = OCR_dataset()
	# dataset.test()