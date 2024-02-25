import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v3 import *
from public_tests import *

'''
numpy is the fundamental package for scientific computing with Python.
matplotlib is a library to plot graphs in Python.
h5py is a common package to interact with a dataset that is stored on an H5 file.
PIL and scipy are used here to test your model with your own picture at the end.
dnn_app_utils provides the functions implemented in the "Building your Deep Neural Network: Step by Step" assignment to this notebook.
np.random.seed(1) is used to keep all the random function calls consistent. It helps grade your work - so please don't change it!
'''

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)