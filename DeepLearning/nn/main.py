import time
import numpy as np
import matplotlib.pyplot as plt
import scipy

# from scipy import ndimage
# from functions import *
from load import *
from dnn_app_utils_v2 import *
# from nn import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x, train_y, test_x, test_y = preprocessing(train_x_orig, train_y, test_x_orig, test_y, classes)

def run_two_layer():
	n_x = 100*100     # num_px * num_px * 3
	n_h = 7
	n_y = 1
	layers_dims = (n_x, n_h, n_y)
	parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 4000, print_cost=True)

	predictions_train = predict(train_x, train_y, parameters)
	predictions_test = predict(test_x, test_y, parameters)

def run_L_layer(layers_dims, num_iterations):
	
	global classes
	parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = num_iterations, print_cost = True)

	pred_train = predict(train_x, train_y, parameters)
	pred_test = predict(test_x, test_y, parameters)


layers_dims = [100*100, 12, 6, 3, 1]
num_iterations = 1500
# run_L_layer(layers_dims, num_iterations)
run_two_layer()

