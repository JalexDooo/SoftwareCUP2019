import os
import cv2
import glob
import math

import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image
from pylab import histogram, interp

import warnings
warnings.filterwarnings('ignore')

from scipy.signal import convolve2d
from scipy.stats import gaussian_kde as kde

class Credit(object):
	"""docstring for Credit"""
	def __init__(self, root):
		super(Credit, self).__init__()
		self.root = root
		self.train_images = glob.glob(self.root+'/test_images/*')

		# 每张图片进行下列操作
		for i in self.train_images:
			

			image = cv2.imread(i, cv2.IMREAD_COLOR)

			image = self.resize(image)



			image = self.blur1(image)
			image = self.gray(image)

			# 旋转矫正
			# img, image = self.rotation(image)

			image = cv2.Canny(image, 50, 200)

			img, image = self.myregion(image)




			
			# img, image = self.region(image)

			# img = self.threshold(img)

			# img = self.edge(img)

			# img, contours, hierarchy = self.rect(img)

			# print(len(hierarchy))

			



			cv2.imshow('test', image)
			cv2.waitKey(0)

			

			'''
			image = Image.imread(i)


			image = self.normalize(image)
			image, image1 = self.hist(image)
			
			d, d1, d2 = self.part(image)
			print(d.shape)
			# print(len(d))
			# image = self.conv2d(image)
			self.myshow(image, d1, d2)
			'''

	def myregion(self, image):
		# print(image.shape)


		
		height, width = image.shape
		# print(height, width)
		bit = int(height/2)
		tens = int(height/5)
		




		image = image[bit-tens:bit+tens, :]
		
		img = image.copy()

		kernel = np.ones((3, 19), np.uint8) # 4, 19
		image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
		image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

		kernel = np.ones((3, 10), np.uint8)

		image = cv2.dilate(image, kernel)

		kernel = np.ones((3, 3), np.uint8)
		image = cv2.erode(image, kernel)


		image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]


		digit_contours = []
		for cnt in contours:
			rect = cv2.minAreaRect(cnt)
			area_width, area_height = rect[1]
			if area_width < area_height:
				area_width, area_height = area_height, area_width
			# wh_ratio = area_width / area_height
			# if wh_ratio > 2 and wh_ratio < 5.5:
			digit_contours.append(rect)
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			img = cv2.drawContours(img, [box], 0, (255, 255, 255), 2)


		# digit_imgs = []

		# for rect in digit_contours:
		# 	if rect[2] > -1 and rect[2] < 1:
		# 		angle = 1
		# 	else:
		# 		angle = rect[2]

		# 	rect = (rect[0], (rect[1][0]+5, rect[1][1]+5), angle)

		# 	print(rect)

		# npimage = np.sum(image, axis=1)

		# print(npimage/255)
		# print('image.shape:',image.shape, ' length:', len(npimage))

		# print(np.where(npimage == np.max(npimage)))

		return image, img
			
	def resize(self, image):
		height, width = image.shape[:2]
		# print(height, width)

		if width >= 1000:
			resize_rate = 1000 / width

			image = cv2.resize(image, (1000, int(height*resize_rate)), interpolation=cv2.INTER_AREA)

		return image



	def gray(self, image):

		return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	def blur1(self, image):

		# return cv2.GaussianBlur(image, (3, 3), 0)
		return cv2.medianBlur(image, 5)

	def region(self, image):
		kernel = np.ones((20, 20), np.uint8)
		# image = image**2
		img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) # OPEN
		img = cv2.addWeighted(image, 1, img, -1, 0)

		return img, image

	def threshold(self, image):
		
		ret, image = cv2.threshold(image, 0, 255, 
			cv2.THRESH_BINARY
			+
			cv2.THRESH_OTSU
			)

		return image
	
	def edge(self, image):
		image = cv2.Canny(image, 50, 200) # 100, 200

		kernel = np.ones((4, 19), np.uint8) # 4, 19
		image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
		image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
		# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
		# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

		return image

	def rect(self, image):

		image, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]

		print('len(contours): ', len(contours))
		# print(contours[0])

		# cv2.drawContours(image, contours, -1, (0, 0, 0), -1)
		# cv2.imshow('drawing', image)




		return image, contours, hierarchy

	def rotation(slef, image):
		

		ret, image = cv2.threshold(image, 0, 255, 
			cv2.THRESH_BINARY
			+
			cv2.THRESH_OTSU
			)

		img = image
		kernel = np.ones((20, 20), np.uint8)
		# image = image**2
		img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) # OPEN
		img = cv2.addWeighted(image, 1, img, -1, 0)


		img = cv2.Canny(img, 50, 200) # 100, 200

		kernel = np.ones((4, 19), np.uint8) # 4, 19
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)





		# img, contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		# c = sorted(contours, key=cv2.contourArea, reverse=True)[1]


		c = np.column_stack(np.where(img > 0))
		
		rect = cv2.minAreaRect(c)

		angle = rect[2]


		box = np.int0(cv2.boxPoints(rect))
		draw_img = cv2.drawContours(image.copy(), [box], -1, (0, 0, 255), 3)
		rows, cols = image.shape[:2]
		M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
		result_img = cv2.warpAffine(image, M, (cols, rows))




		print(angle)
		



		return result_img, image
			
			
			
	###############################################################################################
	def myshow(self, image, d1, d2):

		fig = plt.figure()

		for i in range(len(d2)):
			strr = '33' + str(i+1)
			ax = fig.add_subplot(strr)
			# ax = plt.plot(image)

			img = 255 * (image >= d1[i]) * (image < d1[i+1])
			

			plt.imshow(img, cmap='gray')
			plt.sca(ax)
			# self.hist_show(image)
		plt.show()

		return


	def hist_show(self, image):
		"""
		1. 数据
		2. bins: 直方图长条数目。
		3. normed: 是否将直方图向量归一化，可选项，默认0，
		4. facecolor: 长条形的颜色。
		5. ...
		6. alpha: 透明度。
		"""

		plt.hist(image.flatten(), bins=100, normed=0, facecolor='blue', edgecolor='black', alpha=0.7)
		plt.show()

		return



	def normalize(self, image):
		gray = np.array([0.229, 0.587, 0.114])
		image = np.dot(image, gray)
		# image **=2

		image = ((image - image.min()) / (image.max() - image.min()) * 255).round()

		return image

	def hist(self, image):
		imhist, bins = histogram(image.flatten(), 5, normed=True)
		cdf = imhist.cumsum()
		cdf = cdf*255/cdf[-1]

		image2 = interp(image.flatten(), bins[:5], cdf)
		image2 = image2.reshape(image.shape)

		return image2, bins

	def conv2d(self, image):
		sobel_x = np.c_[
			[1, 1, 1],
			[1, -8, 1],
			[1, 1, 1]
		]

		sobel_y = np.c_[
			[-1, -2, -1],
			[0, 0, 0],
			[1, 2, 1]
		]

		sobel_z = np.c_[
			[-1, 0, 1],
			[-2, 0, 2],
			[-1, 0, 1]
		]
		image = convolve2d(image, sobel_y, mode='same', boundary='symm')
		# image = convolve2d(image, sobel_z, mode='same', boundary='symm')

		return image

	def part(self, image):
		d0 = kde(image.reshape(-1), bw_method=0.2)(range(256))
		d = np.diff(d0)
		d1 = np.where((d[:-1]<0)*(d[1:]>0))[0] #极小值
		d1 = [0]+list(d1)+[256]
		d2 = np.where((d[:-1]>0)*(d[1:]<0))[0] #极大值
		if d1[1] < d2[0]:
			d2 = [0]+list(d2)
		if d1[len(d1)-2] > d2[len(d2)-1]:
			d2 = list(d2)+[255]
		print('d1:', d1)
		print('d2:', d2)
		dc = sum(map(lambda i: d2[i]*(image >= d1[i])*(image < d1[i+1]), range(len(d2))))

		return dc, d1, d2
	###########################################################################################
		



if __name__ == '__main__':
	
	dataset_path = '.'
	test = Credit(dataset_path)

	# image1 = cv2.imread('./duoduo.jpeg', cv2.IMREAD_COLOR)
	# image1 = cv2.resize(image1, (int(image1.shape[1]/4), int(image1.shape[0]/4)), interpolation=cv2.INTER_AREA)


	# image2 = cv2.imread('./dongdong.jpeg', cv2.IMREAD_COLOR)
	# image2 = cv2.resize(image2, (691, 921), interpolation=cv2.INTER_AREA)


	# image = cv2.addWeighted(image2, 0.7, image1, 0.7, 0)

	# # print(image1.shape, image2.shape)

	# cv2.imshow('testt', image)
	# cv2.waitKey(0)







