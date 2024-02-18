# Linear Regression Implementation

This repository contains Python code for implementing linear regression from scratch and validating it against the `sklearn` library. Linear regression is a fundamental technique in statistical modeling for predicting a continuous variable based on one or more predictor variables.

## About the Code

The code consists of a class `linear_regression` which implements linear regression using the closed-form solution. It includes the following functionalities:

- Importing necessary libraries.
- Defining the `linear_regression` class which includes methods for:
  - Splitting the data into training and testing sets.
  - Adding a bias term to the input features.
  - Normalizing the input features.
  - Solving the linear regression using the closed-form solution, both regularized and not regularized.
  - Predicting the output using the trained model.
  - Calculating the Root Mean Square Error (RMSE) and Sum of Square Error (SSE).
  - Fitting the model to the data.
- Fetching the California Housing dataset.
- Creating instances of the `linear_regression` class for both regularized and not regularized models.
- Training the models and printing the results.

## Usage

You can use the provided code as follows:

```python
# Importing the libraries
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Definition of linear_regression class and methods...

# Fetch the data
cal_housing = fetch_california_housing()

# Create an instance of linear_regression class (Not Regularized)
lr = linear_regression(cal_housing.data, cal_housing.target, reg=False, lamda=0.0005)

# Train the model
lr.fit()

# Create an instance of linear_regression class (Regularized)
lr_reg = linear_regression(cal_housing.data, cal_housing.target, reg=True, lamda=0.0005)

# Train the model
lr_reg.fit()
