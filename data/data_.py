import cv2
import glob
import numpy as np

def gray(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if __name__ == '__main__':
	path = './images'
	data_path = glob.glob(path+'/*')
	

	for i in data_path:
		image = cv2.imread(i, cv2.IMREAD_COLOR)
		name = i.split('/')[-1][:7]
		label = i.split('/')[-1][:4]

		image = gray(image)

		imagee = []
		for ii in range(4):
			imagee.append(image[:, 30*ii:30*(ii+1)])


		# print(image1.shape, image2.shape, image3.shape, image4.shape)
		k = 0
		for j in label:
			if j != '_':
				cv2.imwrite('./train_images/'+name+'_'+str(k)+'.png', imagee[k])
			
			k += 1

