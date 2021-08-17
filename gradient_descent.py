'''
Emil Sjöberg

Implementation of gradient descent
'''


import numpy as np
import matplotlib.pyplot as plt
import tabulate
from mpl_toolkits.mplot3d import Axes3D
import random


'''
LINEAR REGRESSION
'''

# x, y
data = np.array([
    [2.4, 1.7], 
    [2.8, 1.85], 
    [3.2, 1.79], 
    [3.6, 1.95], 
    [4.0, 2.1], 
    [4.2, 2.0], 
    [5.0, 2.7]
])

x, y = data[:,0], data[:,1]

#plt.figure(figsize=(4,3))
#plt.scatter(x,y)

# Three functions as linear fit candidates
def f1(x):
	return 0.92*x - 1.0

def f2(x):
	return -0.21*x + 3.4

def f3(x):
	return 0.52*x + 0.1

# Plot the functions
'''
min_x, max_x = min(x), max(x)
fig = plt.figure(figsize=(10,3))

fig.add_subplot(131)
plt.scatter(x, y)
plt.plot([min_x, max_x], [f1(min_x), f1(max_x)])
plt.title('f1')

fig.add_subplot(132)
plt.scatter(x, y)
plt.plot([min_x, max_x], [f2(min_x), f2(max_x)])
plt.title('f2')

fig.add_subplot(133)
plt.scatter(x, y)
plt.plot([min_x, max_x], [f3(min_x), f3(max_x)])
plt.title('f3')
'''
# Error function, sum squared error
def cost(y_pred, y_actual):
	return 0.5*np.sum((y_actual - y_pred)**2)

#fig = plt.figure()
#ax = fig.gca(projection='3d')

# Check all combinations of m between [-2, 4] and b between [-6, 8] to precision of 0.1
M = np.arange(-2, 4, 0.1)
B = np.arange(-6, 8, 0.1)

# Get MSE at every combination
J = np.zeros((len(M), len(B)))
for i, m_ in enumerate(M):
	for j, b_ in enumerate(B):
		J[i][j] = cost(m_*x + b_, y)

# Plot loss surface
'''
B, M = np.meshgrid(B, M)
ax.plot_surface(B, M, J, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
plt.title('Cost for different m and b')
plt.xlabel('b')
plt.ylabel('m')
plt.show()
'''

'''
GRADIENT DESCENT
'''

# Normalizing the data
x = x / np.amax(x, axis=0)
y = y / np.amax(y, axis=0)

# Choose a random initial m and b
m, b = random.random(), random.random()

def F(x, m, b):
	return m*x + b

# Find error
y_pred = F(x, m, b)
init_cost = cost(y_pred, y)

print('Initial parameters: m = {}, b = {}'.format(m, b))
print('Initial cost = {}'.format(init_cost))


# Find partial derivatives of our parameters
def dJdm(x, y, m, b):
	return -np.dot(x, y - F(x, m, b))

def dJdb(x, y, m, b):
	return -np.sum(y - F(x, m, b))

# Choose the learning rate alpha and number of iterations
alpha = 0.01
n_iters = 2000

# Keep track of errors
errors = []
for i in range(n_iters):
	m = m - alpha*dJdm(x, y, m ,b)
	b = b - alpha*dJdb(x, y, m ,b)
	y_pred = F(x, m, b)
	j = cost(y_pred, y)
	errors.append(j)

# Plot cost by iteration
plt.figure(figsize=(16, 3))
plt.plot(range(n_iters), errors, linewidth=2)
plt.title('Cost by iteration')
plt.ylabel('Cost')
plt.xlabel('Iterations')
#plt.show()

# Final error rate
y_pred = F(x, m, b)
final_cost = cost(y_pred, y)

print("Final paremeters: m = {}, b = {}".format(m, b))
print("Final cost = {}".format(final_cost))

# Line of best fit
min_x, max_x = min(x), max(x)

fig= plt.figure(figsize=(3,3))
plt.scatter(x, y)
plt.plot([min_x, max_x], [m*min_x + b, m*max_x + b])
plt.title('Line of best fit')
plt.show()

