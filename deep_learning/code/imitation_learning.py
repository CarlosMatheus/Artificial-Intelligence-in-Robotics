import numpy as np
import matplotlib.pyplot as plt
from utils import sum_gt_zero, xor
from keras import models, layers, losses, optimizers, activations, regularizers, metrics
from math import pi

# Joints' order in the dataset
joints_dict = dict()
joints_dict['leftAnklePitch'] = 0
joints_dict['leftAnkleRoll'] = 1
joints_dict['leftElbowYaw'] = 2
joints_dict['leftHipPitch'] = 3
joints_dict['leftHipRoll'] = 4
joints_dict['leftHipYaw'] = 5
joints_dict['leftKneePitch'] = 6
joints_dict['leftShoulderPitch'] = 7
joints_dict['leftShoulderRoll'] = 8
joints_dict['neckPitch'] = 9
joints_dict['neckYaw'] = 10
joints_dict['rightAnklePitch'] = 11
joints_dict['rightAnkleRoll'] = 12
joints_dict['rightElbowYaw'] = 13
joints_dict['rightHipPitch'] = 14
joints_dict['rightHipRoll'] = 15
joints_dict['rightHipYaw'] = 16
joints_dict['rightKneePitch'] = 17
joints_dict['rightShoulderPitch'] = 18
joints_dict['rightShoulderRoll'] = 19

right_leg_joints = ['rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll']

num_epochs = 30000  # number of epochs for training
# Figure format used for saving figures
fig_format = 'png'
# fig_format = 'svg'
# fig_format = 'eps'

# Loading the dataset
positions = np.loadtxt('positions.txt')
# The dataset contains the walking cycles, but we will use only the first one for training
expected_output = positions[0:40, :]

print(expected_output)

# Creating a input vector (0.008 ms is the sample time of the walking algorithm)
input = np.matrix(0.008 * np.arange(0, expected_output.shape[0])).T

# Setting the random seed of numpy's random library for reproducibility reasons
np.random.seed(0)







alpha = 0.01
lambda_l2 = 0.000  # lambda parameter of the L2 regularization
# lambda_l2 = 0.002  # lambda parameter of the L2 regularizati/on

num_cases = expected_output.shape[0]

# Creates the neural network model in Keras
model = models.Sequential()

# Adds the first layer
# The first argument refers to the number of neurons in this layer
# 'activation' configures the activation function
# input_shape represents the size of the input
# kernel_regularizer configures regularization for this layer
model.add(layers.Dense(75, activation=activations.linear, input_shape=(1,),
                       kernel_regularizer=regularizers.l2(lambda_l2)))
model.add(layers.LeakyReLU(alpha))
model.add(layers.Dense(50, activation=activations.linear, input_shape=(75,),
                       kernel_regularizer=regularizers.l2(lambda_l2)))
model.add(layers.LeakyReLU(alpha))
model.add(layers.Dense(20, activation=activations.linear, input_shape=(50,),
                       kernel_regularizer=regularizers.l2(lambda_l2)))

model.compile(optimizer=optimizers.Adam(), loss=losses.mean_squared_error, metrics=[metrics.binary_accuracy])

history = model.fit(input, expected_output, batch_size=num_cases, epochs=num_epochs)










# input_predict = np.matrix(np.arange(0, input[-1] + 0.001, 0.001)).T

input_predict = input

output = model.predict(input_predict)  # add this line to predict the output from the Neural Network

# output = np.zeros((len(input_predict), np.size(expected_output, 1)))  # remove this line










# Comparing original and copied joint trajectories to evaluate the imitation learning
for joint in right_leg_joints:
    plt.figure()
    plt.plot(input, expected_output[:, joints_dict[joint]] * 180 / pi)
    plt.plot(input_predict, output[:, joints_dict[joint]] * 180.0 / pi)
    plt.grid()
    plt.title(joint)
    plt.xlabel('Time (s)')
    plt.ylabel('Joint Position (°)')
    plt.legend(['Original', 'Neural Network'])
    plt.savefig(joint + '.' + fig_format, format=fig_format)
plt.show()
