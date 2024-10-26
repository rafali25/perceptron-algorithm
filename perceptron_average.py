from numpy import random
import numpy as np

x=random.randint(10, size=(8,3))
d,n = np.shape(x)
theta = np.zeros((d,1))
theta_not = np.zeros((1,1))
labels = [1,-1,1]
theta_sum = np.zeros((d,1))
theta_not_sum = np.zeros((1,1))
T = 100
for t in range(T):
    for i in range(n):
        data_point = x[:,i]
        print(data_point)
        data_point_to_be_added_to_theta = np.array([data_point])
        activation_value = np.sign(np.dot(data_point, theta)+theta_not)
        if activation_value == 0:
            activation_value = -1
        if activation_value != labels[i]:
            theta = theta + labels[i]*data_point_to_be_added_to_theta.T
            theta_not = theta_not + labels[i]
        theta_sum += theta
        theta_not_sum += theta_not
    print(theta_sum/(n*T),theta_not_sum/(n*T))
print(x)