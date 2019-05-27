import numpy as np




if __name__ == '__main__':

	x = np.arange(32).reshape((8, 4))
	print (x[np.ix_([1,5,7,2],[0,3,1,2])])
	print(x)