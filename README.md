# Perceptron Algorithm Example

This repository contains an implementation of the Perceptron algorithm using `numpy`. The code initializes random data, trains a simple Perceptron model, and updates its parameters based on the misclassified points during each iteration.

## Overview

The Perceptron is a type of linear classifier that updates its weights based on the classification error. It is an iterative algorithm that adjusts the decision boundary by modifying the weights after each incorrect classification.

## Code Explanation

### Libraries
- `numpy`: For handling arrays and numerical computations.
- `random`: For generating random integer data points.

### Main Components
1. **Data Initialization**:
   - Random integer data `x` is generated using `random.randint()`, producing an 8x3 matrix.
   - `d, n = np.shape(x)` extracts the dimensions of the matrix: `d` (number of features) and `n` (number of samples).

2. **Parameters Initialization**:
   - `theta`: A zero vector initialized to store the weights of the model (size = `d x 1`).
   - `theta_not`: A zero scalar bias term.
   - `labels`: A simple list of labels for the data points, where class 1 and -1 represent two possible categories.
   - `theta_sum` and `theta_not_sum`: Variables for averaging weights and bias over multiple iterations.

3. **Training Loop**:
   - The code runs for `T = 100` iterations. In each iteration:
     - The algorithm iterates through each data point `x[:, i]`.
     - It calculates the activation value using the dot product of `theta` and the data point.
     - If the data point is misclassified, `theta` and `theta_not` are updated accordingly.
     - The sum of `theta` and `theta_not` over iterations is accumulated.

4. **Output**:
   - After each iteration, the average weights `theta_sum/(n*T)` and bias `theta_not_sum/(n*T)` are printed.
   - At the end, the randomly generated data matrix `x` is printed.

### git clone 
```bash
https://github.com/your-username/perceptron-algorithm-example.git
