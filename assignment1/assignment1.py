import numpy
import helper
import math
import matplotlib.pyplot
import random

### Common data for all the assignment
training_csv = "2018s1-mo444-assignment-01/train.csv"
test_csv = "2018s1-mo444-assignment-01/test.csv"
test_target_csv = "2018s1-mo444-assignment-01/test_target.csv"

# Ignores the first 2 non-predictive features
training_feature_set = numpy.loadtxt (open (training_csv, "rb"), delimiter = ",", skiprows = 1, usecols = range (2,60)).transpose ()
print ("Training set matrix size = " + str(training_feature_set.shape))

training_target_set = numpy.loadtxt (open (training_csv, "rb"), delimiter = ",", skiprows = 1, usecols = 60)
print ("Training target set matrix size = " + str(training_target_set.shape))

test_feature_set = numpy.loadtxt (open (test_csv, "rb"), delimiter=",", skiprows = 1, usecols = range (2,60)).transpose ()
print ("Test set matrix size = " + str(test_feature_set.shape))

test_target_set = numpy.loadtxt (open (test_target_csv, "rb"), delimiter=",", skiprows = 1)
print ("Test target set matrix size = " + str(test_target_set.shape))

### Linear regression Y = 0oXo + 01X1 + 02X2 + ...., which Xo = 1 and X1, X2, ..., Xn are the features

print ("Linear Regression - Y = 0oXo + 01X1 + 02X2 + ...")

lr_results = helper.linearRegression ("data/lr.npy", training_feature_set, training_target_set, test_feature_set, test_target_set)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_results[1])), lr_results[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_results[2])), lr_results[2], label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle ("Linear Regression")

matplotlib.pyplot.grid ()

### Normal Form 0 = (XtX)^1X^tY

print ("Normal Form 0 = (XtX)^1X^tY")
norm_results = helper.normalEquation ("data/norm-lr.npy", training_feature_set, training_target_set, test_feature_set, test_target_set, lr_results[0])

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (norm_results[1], norm_results[2], label = "Training RMS")
test_plot, = ax.plot (norm_results[1], norm_results[3], label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# training samples")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$\theta = (X^TX)^{-1}X^TY$")
matplotlib.pyplot.suptitle ("Normal Equation")
matplotlib.pyplot.grid ()

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)
ax.plot (norm_results[1], norm_results[4])
matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# training samples")
matplotlib.pyplot.title (r"$\theta = (X^TX)^{-1}X^TY$")
matplotlib.pyplot.suptitle (r"RMS between LR and Normal Equations $\theta$")
matplotlib.pyplot.grid ()

### Linear regression Y = 0oXo + 01X1 + 02X2 + ...., which Xo = 1 and X1, X2, ..., Xn are the features with REGULARIZATION

regularization = 0.8

print ("Linear Regression - Y = 0oXo + 01X1 + 02X2 + ...")

lr_results = helper.linearRegression ("data/lr-reg.npy", training_feature_set, training_target_set, test_feature_set, test_target_set, regularization = regularization)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_results[1])), lr_results[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_results[2])), lr_results[2], label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle ("Regularized linear Regression with $\lambda = " + str(regularization) + "$")

matplotlib.pyplot.grid ()

### Normal Form 0 = (XtX)^1X^tY with REGULARIZATION

regularization = 0.8

print ("Normal Form 0 = (XtX + lambdaI)^1X^tY")
norm_results = helper.normalEquation ("data/norm-lr-reg.npy", training_feature_set, training_target_set, test_feature_set, test_target_set, lr_results[0], regularization = regularization)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (norm_results[1], norm_results[2], label = "Training RMS")
test_plot, = ax.plot (norm_results[1], norm_results[3], label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# training samples")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$\theta = (X^TX+ \lambda I)^{-1}X^TY$")
matplotlib.pyplot.suptitle ("Regularized Normal Equation")
matplotlib.pyplot.grid ()

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)
ax.plot (norm_results[1], norm_results[4])
matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# training samples")
matplotlib.pyplot.title (r"$\theta = (X^TX + \lambda I)^{-1}X^TY$")
matplotlib.pyplot.suptitle (r"RMS between LR and Normal Equations $\theta$")
matplotlib.pyplot.grid ()

# More complex models

matplotlib.pyplot.show (block = True)
