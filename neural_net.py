'''
Emil Sjöberg

Implementation of a neural network
'''


import numpy as np
import matplotlib.pyplot as plt
import tabulate
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.datasets import load_iris
import itertools

'''
NEURAL NETWORKS
'''

iris = load_iris()
data, labels = iris.data[:,0:2], iris.data[:,2]

num_samples = len(labels) # Size of dataset

# Shuffle the dataset
shuffle_order = np.random.permutation(num_samples)
data = data[shuffle_order, :]
labels = labels[shuffle_order]

def weighted_sum(x, w, b):
	return b + np.dot(w, x)

# Set parameters
w = [0.2, 0.6]
b = -0.3

# For example, use the first data point
X, y = data, labels

pred_y = [weighted_sum(x, w, b) for x in X]

#print("For x = [{}, {}], predicted = {}, actual = {}".format(X[0][0], X[0][1], pred_y[0], y[0]))

# Sum squared error
def cost(pred_y, y_actual):
	return 0.5*np.sum((y_actual - pred_y)**2)

error = cost(pred_y, y)
#print("Error = ", error)

# Gradient descent

# Normalize data
X = X / np.amax(X, axis=0)
y = y / np.amax(y, axis=0)

# Get random inital w and b
w, b = [random.random(), random.random()], random.random()

# Create function w1*x1 + w2*x2 + b
def F(X, w, b):
	return np.sum(w*X, axis=1) + b

# Get error
y_pred = F(X, w, b)
init_cost = cost(y_pred, y)

#print("Inital parameters, w1 = {}, w2 = {}, b = {}".format(w[0], w[1], b))
#print("Inital cost = ", init_cost)

# Partial derivatives of our parameters
def dJdw1(X, y, w, b):
	return -np.dot(X[:,0], y - F(X, w, b))

def dJdw2(X, y, w, b):
	return -np.dot(X[:,1], y - F(X, w, b))

def dJdb(X, y, w, b):
	return -np.sum(y - F(X, w, b))

# Choose alpha and number of iterations
alpha = 0.001
n_iters = 2000

# Run through gradient descent
errors = []
for i in range(n_iters):
	w[0] = w[0] - alpha*dJdw1(X, y, w, b)
	w[1] = w[1] - alpha*dJdw2(X, y, w, b)
	b = b - alpha*dJdb(X, y, w, b)
	y_pred = F(X, w, b)
	j = cost(y_pred, y)
	errors.append(j)

# Plot error
'''
plt.figure(figsize=(16,3))
plt.plot(range(n_iters), errors, linewidth=2)
plt.title('Cost by iteration')
plt.ylabel('Cost')
plt.xlabel('Iterations')
'''

# Final error rate
y_pred = F(X, w, b)
final_cost = cost(y_pred, y)

#print("Final parameters: w1 = {}, w2 = {}, b = {}".format(w[0], w[1], b))
#print("Final cost = ", final_cost)

#plt.show()

# Sigmoid function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# Reset our parameters
w = [0.2, 0.6]
b = -0.3

X, y = data, labels

# Get weighted sum
Z = [weighted_sum(x, w, b) for x in X]

# Transform the weighted sum with a sigmoid
y_pred = [sigmoid(z) for z in Z]

# Evaluate error
error = cost(y_pred, y)

# Graph of our neurons activity
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
x1, x2 = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
y = sigmoid(w[0]*x1 + w[1]*x2 + b)
ax.plot_surface(x1, x2, y, rstride=1, cstride=1, cmap=plt.cm.coolwarm, antialiased=False)
plt.show()
'''

W1 = np.random.randn(2, 3)
W2 = np.random.randn(3, 1)

# First layer weighted sum
z = np.dot(X, W1)

# Put it through sigmoid
z = sigmoid(z)

# Do another dot product at the end
y_pred = np.dot(z, W2)

# What is our cost
error = cost(y_pred, y)

# Class defining a neural network
class Neural_Network(object):
	def __init__(self, n0, n1, n2):
		self.n0 = n0
		self.n1 = n1
		self.n2 = n2

		# Initialize weights
		self.W1 = np.random.randn(self.n0, self.n1)
		self.W2 = np.random.randn(self.n1, self.n2)

	def predict(self, x):
		z = np.dot(x, self.W1)
		z = sigmoid(z)
		y = np.dot(z, W2)
		return y

# Initialize neural network with 2 input neurons, 3 hidden layers and 1 outpu neuron
net = Neural_Network(2, 3, 1)

# Run networks predict function
X, y = data, labels
y_pred = net.predict(X)
error = cost(y_pred, y)

#print("Predicted = {}, Actual = {}, Error = {}".format(pred_y[0], y[0], error))

# Numerical method for calculating the gradient
def get_gradient(net, X, y):
	w_delta = 1e-8

	# Get the current value of the loss
	y_pred_current = net.predict(X)
	error_current = cost(y_pred_current, y)

	# Grab the current weights and copy them
	dw1, dw2 = np.zeros((net.n0, net.n1)), np.zeros((net.n1, net.n2))
	W1, W2 = np.copy(net.W1), np.copy(net.W2)

	# For the first layer, iterate through each weight,
	# Perturb(stör), it slightly and calculate the numerical
	# slope between that loss and the original loss
	for i,j in itertools.product(range(net.n0), range(net.n1)):
		net.W1 = np.copy(W1)
		net.W1[i][j] += w_delta
		y_pred = net.predict(X)
		error = cost(y_pred, y)
		dw1[i][j] = (error - error_current)/w_delta

	# Do the same thing for the second layer
	for i,j in itertools.product(range(net.n1), range(net.n2)):
		net.W2 = np.copy(W2)
		net.W2[i][j] += w_delta
		y_pred = net.predict(X)
		error = cost(y_pred, y)
		dw2[i][j] = (error - error_current)/w_delta

	# Restore the original weights
	net.W1, net.W2 = np.copy(W1), np.copy(W2)

	return dw1, dw2

'''
TRAIN NEURAL NETWORK
Load our dataset, instantiate neural network, train it on the data using the gradient method
'''

# Load data and labels
X, y = data, labels.reshape((len(labels), 1))

# Normalize data
X = X / np.amax(X, axis=0)
y = y / np.amax(y, axis=0)

# Create a neural net with 2 input layers, 3 hidden layers and 1 output layer
net = Neural_Network(2, 3, 1)

# Get the current cost
y_orig = net.predict(X)
init_cost = cost(y_orig, y)
print("Initial cost = ", init_cost)

# Set the learning rate and how many epochs(updates) to try
learning_rate = 0.01
n_epochs = 2000

# For each epoch, calculate the gradient, then subtract it from the parameters, and save the cost
errors = []
for i in range(n_epochs):
	dw1, dw2 = get_gradient(net, X, y)
	net.W1 = net.W1 - learning_rate*dw1
	net.W2 = net.W2 - learning_rate*dw2
	y_pred = net.predict(X)
	error = cost(y_pred, y)
	errors.append(error)

plt.plot(range(0, len(errors)), errors)

# Get final cost
y_pred = net.predict(X)
final_cost = cost(y_pred, y)
print("Final cost = ", final_cost)
plt.show()