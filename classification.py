'''
Emil Sjöberg

Example of classification in ML
'''


import numpy as np
import matplotlib.pyplot as plt
import tabulate

'''
CLASSIFICATION
'''

# K-NEAREST NEIGHBORS
from sklearn.datasets import load_iris

# Load iris and grab our data
iris = load_iris()
labels, data = iris.target, iris.data

num_samples = len(labels) # Size of dataset
num_features = len(iris.feature_names) # Number of columns/variables

# Shuffle the dataset
shuffle_order = np.random.permutation(num_samples)
data = data[shuffle_order, :]
labels = labels[shuffle_order]

# Table with first 20 samples
label_names = np.array([iris.target_names[l] for l in labels])
table_labels = np.array(['Class'] + iris.feature_names).reshape((1, 1 + num_features))
class_names = iris.target_names
table_data = np.concatenate([np.array(label_names).reshape(num_samples, 1), data], axis=1)[0:20]

# Display table
table_full = np.concatenate([table_labels, table_data], axis=0)
#print(tabulate.tabulate(table_full))

# Plot data
x, y, lab = data[:, 0], data[:, 1], labels
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=lab)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris dataset')

# Add new point
new_x, new_y = 6.5, 3.7
plt.scatter(new_x, new_y, c='red', cmap=None, edgecolor='k')

# Calculate the distance between the new point and each of the points in our labeled dataset
distances = np.sum((data[:, 0:2] - [new_x, new_y])**2, axis=1)

# Find the index of the point whose distance is closest
closest_point = np.argmin(distances)

# Take it's label
new_label = labels[closest_point]

print('Predicted label: {}'.format(new_label))
