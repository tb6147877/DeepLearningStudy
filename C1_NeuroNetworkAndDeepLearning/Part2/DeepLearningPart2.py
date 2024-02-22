### v1.1

"""
numpy is the fundamental package for scientific computing with Python.
sklearn provides simple and efficient tools for data mining and data analysis.
matplotlib is a library for plotting graphs in Python.
testCases provides some test examples to assess the correctness of your functions
planar_utils provide various useful functions used in this assignment
"""

# Package imports
import numpy as np
import copy
import pylab
import matplotlib.pyplot as plt
from testCases_v2 import *
from public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# Load data
X, Y = load_planar_dataset()

# Visualize the data:
# Visualize the dataset using matplotlib. The data looks like a "flower" with some red (label y=0) and some blue (y=1) points.
# Your goal is to build a model to fit this data. In other words, we want the classifier to define regions as either red or blue.
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
pylab.show()

# How many training examples do you have? In addition, what is the shape of the variables X and Y?
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size
print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

'''
Before building a full neural network, let's check how logistic regression performs on this problem. 
You can use sklearn's built-in functions for this. Run the code below to train a logistic regression classifier on the dataset.
'''
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
pylab.show()
# Print accuracy
LR_predictions = clf.predict(X.T)
value1 = np.dot(Y,LR_predictions.reshape((400, 1)))[0][0]
value2 = np.dot(1-Y,1-LR_predictions.reshape((400, 1)))[0][0]
print ('Accuracy of logistic regression: %d ' % float((value1 + value2)/float(Y.size)*100) + '% ' + "(percentage of correctly labelled datapoints)")
#  The dataset is not linearly separable, so logistic regression doesn't perform well. Hopefully a neural network will do better. Let's try this now!





