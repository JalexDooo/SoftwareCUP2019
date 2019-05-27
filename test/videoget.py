import cv2
import time

# 1PD Outdoor: 10.42.25.238
# 1PD Indoor: 10.42.25.120


class GetVideoTest(object):
	"""docstring for GetVideoTest"""
	def __init__(self):
		self.PD1_Outdoor = '10.42.25.238'
		self.PD1_Indoor = '10.42.25.120'
		self.url = 'rtsp://admin:ddxxzx123@'+self.PD1_Outdoor+':554/Streaming/Channels/1'
		print(self.url)
		self.VideoGet(self.url)

	def VideoGet(self, url):
		cap = cv2.VideoCapture(url)

		aa, bb, cc = (cv2.__version__).split('.')

		if int(aa) < 3:
			fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
			print(fps)
		else :
			fps = cap.get(cv2.CAP_PROP_FPS)
			print(fps)



		imwrite = 'D:/Algorithm/video'
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		out = cv2.VideoWriter(imwrite+'/Scene1Morning'+'.avi', fourcc, 25.0, (1280, 720))

		
		while True:
			ret, frame = cap.read()
			if ret == False:
				print('Frame continue...')
				continue
			# print(frame.shape)

			cv2.imshow('test', frame)
			out.write(frame)
			# cv2.imwrite(imwrite+'/test.png', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		cap.release()
		return 



if __name__ == '__main__':
	test = GetVideoTest()

	# st = time.time()
	# i = 0
	# while True:
	# 	i += 1
	# 	if time.time() - st >= 1:
	# 		print(i)
	# 		break
	

