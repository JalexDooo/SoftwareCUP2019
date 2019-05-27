import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras.layers import Dense, Dropout, LSTM
from keras.layers import Activation, Flatten, Input, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

file_root = '../housenumbers'

housenumbers = glob.glob(file_root+'/*')
print(housenumbers)

train_images = pd.read_csv(file_root+'/train_images.csv')
train_labels = pd.read_csv(file_root+'/train_labels.csv')

test_images = pd.read_csv(file_root+'/test_images.csv')
test_labels = pd.read_csv(file_root+'/test_labels.csv')

extra_images = pd.read_csv(file_root+'/extra_images.csv')
extra_labels = pd.read_csv(file_root+'/extra_labels.csv')

# print(train_images.ix[:10, :10])
train_images = train_images.ix[:, 1:].as_matrix().astype('float32')
train_labels = train_labels.ix[:, 1:].as_matrix().astype('int16')

test_images = test_images.ix[:, 1:].as_matrix().astype('float32')
test_labels = test_labels.ix[:, 1:].as_matrix().astype('int16')

extra_images = extra_images.ix[:, 1:].as_matrix().astype('float32')
extra_labels = extra_labels.ix[:, 1:].as_matrix().astype('int16')

print('Label: ', train_labels[100])
plt.imshow(train_images[100].reshape(32, 32), cmap=plt.cm.bone)
plt.show()




