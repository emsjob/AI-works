'''
Emil Sjöberg

Solving the cartpole problem fromm OpenAI gym with reinforcement learning
'''
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.utils import to_categorical

import os
import numpy as np
import time
import math
import pandas as pd
import cv2
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from numpy import argmax

import matplotlib
import matplotlib.pyplot as plt

import random
import gym

# Helper function
def running_mean(x, N=10):
    """ Return the running mean of N element in a list
    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# Parameters
EPISODES       = 15000  # Number of eposides to run, org val = 15000
EPSILON        = 0.5    # Chance to explore (take a radom step instead of an "optimal" one), org val = 0.1
GAMMA          = 0.5    # How much previous steps should be rewarded now, discount rate, org val = 0.9
LEARNING_RATE  = 0.25   # Learning rate, org val = 0.1
DISCRETE_STEPS = 12    # Discretization steps per state variable (avoid odd numbers), org val = 10

# Variables for epsilon decay
epsilon_decay_start = 1
epsilon_decay_end = EPISODES//2
epsilon_decay_value = EPSILON/(epsilon_decay_end - epsilon_decay_start)

# Variables for learning rate decay
learning_decay_start = 1
learning_decay_end = EPISODES//2
learning_decay_value = LEARNING_RATE/(learning_decay_end - learning_decay_start)


def make_state(observation):
  low = [-4.8, -10., -0.41888, -10.] # ( changed -0.41 into more correct -0.41888 for 24 deg.)
  high = [4.8, 10., 0.41888, 10.]
  state = 0

  for i in range(4):
      # State variable, projected to the [0, 1] range
      state_variable = (observation[i] - low[i]) / (high[i] - low[i])

      # Discretize. A variable having a value of 0.53 will lead to the integer 5,
      # for instance.
      state_discrete = int(state_variable * DISCRETE_STEPS)
      state_discrete = max(0, state_discrete) # should not be needed
      state_discrete = min(DISCRETE_STEPS-1, state_discrete)

      state *= DISCRETE_STEPS
      state += state_discrete
      # Make state into a 4 "digit" number (between 0 and 9999, if 10 discrete steps)
  return state

episode_reward = np.zeros(EPISODES)

# Create the Gym environment (CartPole)
env = gym.make('CartPole-v1')

print('Action space is:', env.action_space)
print('Observation space is:', env.observation_space)

# Q-table for the discretized states, and two actions
num_states = DISCRETE_STEPS ** 4
qtable = [[0., 0.] for state in range(num_states)]
print('Q-table = %.0f x %.0f' % (len(qtable),len(qtable[0]) ))

# Initialize total return
total_return = 0.0

for i in range(EPISODES):
  state4D = env.reset()
  state = make_state(state4D)

  terminate = False
  cumulative_reward = 0.0

  # Initialize episode return
  episode_return = 0.0

  # Loop over time-steps
  while not terminate:

    qvalues = qtable[state]
    greedy_action = np.argmax(qvalues)
    # Get action
    if random.random() < EPSILON:
      action = random.randrange(2)
    else:
      action = greedy_action
    
    # Perform the action
    next_state, reward, terminate, info = env.step(action) # Info is ignored
    next_state = make_state(next_state)

    # Update the Q-table
    td_error = reward + GAMMA * max(qtable[next_state]) - qtable[state][action]
    qtable[state][action] += LEARNING_RATE * td_error

    # Update statistics
    cumulative_reward += reward
    state = next_state

    episode_return += reward # Increment episode return
  total_return += episode_return # Increment total return

  # Store reward for every episode
  episode_reward[i] = cumulative_reward


  # Decaying is done every run if run number is within decay rnge
  if epsilon_decay_start <= i <= epsilon_decay_end:
    EPSILON -= epsilon_decay_value
    EPSILON = abs(EPSILON)
  
  if learning_decay_start <= i <= learning_decay_end:
    LEARNING_RATE -= learning_decay_value
    LEARNING_RATE = abs(LEARNING_RATE)

  # Per-episode statistics
  if ((i % 500) == 0):
    print('Episode: ', i, ', Reward:', cumulative_reward)
    #print('Epsilon: ', EPSILON)
    #print('Learning rate: ', LEARNING_RATE)

# Average reward for the last episodes
avg_reward = sum(episode_reward[-5000:])/len(episode_reward[-5000:])
print("Average reward: ", avg_reward)

# Exploration of Q-table:
filled = []
for i in qtable:
  if i == [0.0, 0.0]:
    filled.append(i)

print('Size of filled Q-table: ', len(filled))

# Plot graph
plt.figure(figsize=(16, 4))
plt.plot(episode_reward,"b")
y_av = running_mean(episode_reward, N=100)
plt.plot(y_av,"r")
plt.show()

# Print out average return
avg_return = total_return/EPISODES
print("Average return = ", avg_return)
