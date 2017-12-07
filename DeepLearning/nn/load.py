import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


from nn import *

def load_data():

	train_x_pos_path = 'data/training/pos'
	train_x_neg_path = 'data/training/neg'

	test_x_pos_path = 'data/testing/pos'
	test_x_neg_path = 'data/testing/neg'

	train_x_pos = read_images(train_x_pos_path)
	train_x_neg = read_images(train_x_neg_path)

	train_y_pos = [1 for i in range(len(train_x_pos))]
	train_y_neg = [0 for i in range(len(train_x_neg))]

	train_x = np.array(train_x_pos + train_x_neg)
	train_y = np.array(train_y_pos + train_y_neg)

	test_x_pos = read_images(test_x_pos_path)
	test_x_neg = read_images(test_x_neg_path)

	test_y_pos = [1 for i in range(len(test_x_pos))]
	test_y_neg = [0 for i in range(len(test_x_neg))]

	test_x = np.array(test_x_pos + test_x_neg)
	test_y = np.array(test_y_pos + test_y_neg)

	classes = np.array(['non-face', 'face'])

	train_y = np.reshape(train_y, (1, len(train_y)))
	test_y = np.reshape(test_y, (1, len(test_y)))

	return train_x, train_y, test_x, test_y, classes


def read_images(path):

	res =  []
	for file in os.listdir(path):

		try:
			img = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
			res.append(img.copy())
		except:
			continue

	return res


def preprocessing(train_x_orig, train_y, test_x_orig, test_y, classes=None):

	# train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

	# print train_x_orig.shape


	m_train = train_x_orig.shape[0]
	m_test = test_x_orig.shape[0]

	num_px = train_x_orig.shape[1]

	print ("Number of training examples: " + str(m_train))
	print ("Number of testing examples: " + str(m_test))
	print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
	print ("train_x_orig shape: " + str(train_x_orig.shape))
	print ("train_y shape: " + str(train_y.shape))
	print ("test_x_orig shape: " + str(test_x_orig.shape))
	print ("test_y shape: " + str(test_y.shape))

	# reshape and standardize the images before feeding to the network

	train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
	test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

	# Standardize data to have feature values between 0 and 1.
	train_x = train_x_flatten/255.
	test_x = test_x_flatten/255.

	print ("train_x's shape: " + str(train_x.shape))
	print ("test_x's shape: " + str(test_x.shape))

	return train_x, train_y, test_x, test_y