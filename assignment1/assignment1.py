import numpy
import helper
import math
import matplotlib.pyplot
import random
import sys

matplotlib.pyplot.style.use('grayscale')

### Common data for all the assignment
training_csv = "2018s1-mo444-assignment-01/train.csv"
test_csv = "2018s1-mo444-assignment-01/test.csv"
test_target_csv = "2018s1-mo444-assignment-01/test_target.csv"

# Ignores the first 2 non-predictive features
training_feature_set = numpy.loadtxt (open (training_csv, "rb"), delimiter = ",", skiprows = 1, usecols = range (2,60)).transpose ()
print ("Training set matrix size = " + str(training_feature_set.shape))

training_target_set = numpy.loadtxt (open (training_csv, "rb"), delimiter = ",", skiprows = 1, usecols = 60)
print ("Training target set matrix size = " + str(training_target_set.shape))

print (training_feature_set.shape)
print (training_target_set.shape)

test_feature_set = numpy.loadtxt (open (test_csv, "rb"), delimiter=",", skiprows = 1, usecols = range (2,60)).transpose ()
print ("Test set matrix size = " + str(test_feature_set.shape))

test_target_set = numpy.loadtxt (open (test_target_csv, "rb"), delimiter=",", skiprows = 1)
print ("Test target set matrix size = " + str(test_target_set.shape))

# training_feature_set = helper.processDiscreteFeatures (training_feature_set, discrete_groups = [range (11, 17), range (29, 37)])
# test_feature_set = helper.processDiscreteFeatures (test_feature_set, discrete_groups = [range (11, 17), range (29, 37)])

print (training_feature_set.shape)
print (test_feature_set.shape)
### Linear regression Y = 0oXo + 01X1 + 02X2 + ...., which Xo = 1 and X1, X2, ..., Xn are the features

print ("Linear Regression - Y = 0oXo + 01X1 + 02X2 + ...")

lr_results = helper.linearRegression ("data/lr.npy", training_feature_set, training_target_set, \
                                                     test_feature_set, test_target_set)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_results[1])), lr_results[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_results[2])), lr_results[2], "--", label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle ("Linear regression RMS errors")

matplotlib.pyplot.grid ()
### Normal Form 0 = (XtX)^1X^tY

print ("Normal Form 0 = (XtX)^1X^tY")
norm_results = helper.normalEquation ("data/norm-lr.npy", training_feature_set, training_target_set, test_feature_set, test_target_set, lr_results[0])

fig, ax = matplotlib.pyplot.subplots (2)

traning_plot, = ax[0].plot (norm_results[1], norm_results[2], label = "Training RMS")
test_plot, = ax[0].plot (norm_results[1], norm_results[3], "--", label = "Testing RMS")

ax[0].set_ylabel ("Root Mean Square Error")
ax[0].set_xlabel ("# training samples")
ax[0].legend (handles = [traning_plot, test_plot])
ax[0].set_title  ("Normal equation RMS errors")
ax[0].grid ()

ax[1].plot (norm_results[1], norm_results[4])
ax[1].set_ylabel ("Root Mean Square Error")
ax[1].set_xlabel ("# training samples")
ax[1].set_title (r"RMS between LR and normal equations $\theta$ arrays")
ax[1].grid ()

matplotlib.pyplot.tight_layout()

print ("Minimum RMS with Normal Equation = " + str (min (norm_results[3])))
print ("Minimum RMS with LR = " + str (min (lr_results[2])))

### Linear regression Y = 0oXo + 01X1 + 02X2 + ...., which Xo = 1 and X1, X2, ..., Xn are the features with REGULARIZATION

regularization = 1

print ("Linear Regression - Y = 0oXo + 01X1 + 02X2 + ...")

lr_reg_results = helper.linearRegression ("data/lr-reg.npy", training_feature_set, training_target_set, \
                                                             test_feature_set, test_target_set, \
                                                             regularization = regularization)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_reg_results [1])), lr_reg_results[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_reg_results [2])), lr_reg_results [2], "--", label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle ("Regularized linear regression with $\lambda = " + str(regularization) + "$ RMS errors")

matplotlib.pyplot.grid ()

regularization = 20

print ("Linear Regression - Y = 0oXo + 01X1 + 02X2 + ...")

lr_reg_results_4 = helper.linearRegression ("data/lr-reg-4.npy", training_feature_set, training_target_set, \
                                                                test_feature_set, test_target_set, \
                                                                regularization = regularization)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_reg_results_4 [1])), lr_reg_results_4[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_reg_results_4 [2])), lr_reg_results_4 [2], "--", label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle ("Regularized linear regression with $\lambda = " + str(regularization) + "$ RMS errors")

matplotlib.pyplot.grid ()

### Compare results
fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_results[1])), lr_results[1], ".", label = r"Training RMS - $\lambda = 0.0$")
traning_plot_reg, = ax.plot (range(len (lr_reg_results[1])), lr_reg_results[1], label = r"Training RMS - $\lambda = 1.0$")
test_plot_reg, = ax.plot (range(len (lr_reg_results[2])), lr_reg_results[2], "--", label = r"Testing RMS - $\lambda = 1.0$")
traning_plot_4, = ax.plot (range(len (lr_reg_results_4[1])), lr_reg_results_4[1], "-.", label = r"Training RMS - $\lambda = 20.0$")
test_plot_4, = ax.plot (range(len (lr_reg_results_4[2])), lr_reg_results_4[2], label = r"Testing RMS - $\lambda = 20.0$")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, traning_plot_reg, test_plot_reg, traning_plot_4, test_plot_4])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle (r"Regularized linear regression with gradient descent - comparison between different $\lambda$")

matplotlib.pyplot.grid ()

### Normal Form 0 = (XtX)^1X^tY with REGULARIZATION

regularization = 1

print ("Normal Form 0 = (XtX + lambdaI)^1X^tY")
norm_reg_results = helper.normalEquation ("data/norm-lr-reg.npy", training_feature_set, training_target_set, test_feature_set, test_target_set, lr_reg_results[0], regularization = regularization)

fig, ax = matplotlib.pyplot.subplots (2)

traning_plot, = ax[0].plot (norm_reg_results[1], norm_reg_results[2], label = "Training RMS")
test_plot, = ax[0].plot (norm_reg_results[1], norm_reg_results[3], "--", label = "Testing RMS")

ax[0].set_ylabel ("Root Mean Square Error")
ax[0].set_xlabel ("# training samples")
ax[0].legend (handles = [traning_plot, test_plot])
ax[0].set_title  ("Normal equation with regularization ($\lambda = " + str(regularization) + "$)")
ax[0].grid ()

ax[1].plot (norm_reg_results[1], norm_reg_results[4])
ax[1].set_ylabel ("Root Mean Square Error")
ax[1].set_xlabel ("# training samples")
ax[1].set_title (r"RMS between LR and normal equations $\theta$ arrays with regularization")
ax[1].grid ()

matplotlib.pyplot.tight_layout()

regularization = 20.0

print ("Normal Form 0 = (XtX + lambdaI)^1X^tY")
norm_reg_results_4 = helper.normalEquation ("data/norm-lr-reg-4.npy", training_feature_set, training_target_set, test_feature_set, test_target_set, lr_reg_results_4[0], regularization = regularization)

fig, ax = matplotlib.pyplot.subplots (2)

traning_plot, = ax[0].plot (norm_reg_results_4[1], norm_reg_results_4[2], label = "Training RMS")
test_plot, = ax[0].plot (norm_reg_results_4[1], norm_reg_results_4[3], "--", label = "Testing RMS")

ax[0].set_ylabel ("Root Mean Square Error")
ax[0].set_xlabel ("# training samples")
ax[0].legend (handles = [traning_plot, test_plot])
ax[0].set_title  ("Normal equation with regularization ($\lambda = " + str(regularization) + "$)")
ax[0].grid ()

ax[1].plot (norm_reg_results_4 [1], norm_reg_results_4 [4])
ax[1].set_ylabel ("Root Mean Square Error")
ax[1].set_xlabel ("# training samples")
ax[1].set_title (r"RMS between LR and normal equations $\theta$ arrays with regularization")
ax[1].grid ()

matplotlib.pyplot.tight_layout()

print ("Minimum RMS with Normal Equation (Regularized) = " + str (min (norm_reg_results[3])))
print ("Minimum RMS with LR (Regularized) = " + str (min (lr_reg_results[2])))

### Compare results
fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot_reg, = ax.plot (range(len (norm_reg_results[1])), norm_reg_results[2], label = r"Training RMS - $\lambda = 1.0$")
test_plot_reg, = ax.plot (range(len (norm_reg_results[1])), norm_reg_results[3], "--", label = r"Testing RMS - $\lambda = 1.0$")
traning_plot_4, = ax.plot (range(len (norm_reg_results_4[1])), norm_reg_results_4[2], "-.", label = r"Training RMS - $\lambda = 20.0$")
test_plot_4, = ax.plot (range(len (norm_reg_results_4[1])), norm_reg_results_4[3], label = r"Testing RMS - $\lambda = 20.0$")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# training samples")
matplotlib.pyplot.legend(handles = [traning_plot_reg, test_plot_reg, traning_plot_4, test_plot_4])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle (r"Regularized linear regression with normal equations\nComparison between different $\lambda$")

matplotlib.pyplot.grid ()

# Variable selection

corr = helper.corr ("data/correlation.npy", training_feature_set, training_target_set)
sorted_index = numpy.argsort (corr)

sorted_corr = []
for i in sorted_index:
    sorted_corr.append (corr[i])

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)
labels = [str(i) for i in sorted_index]
matplotlib.pyplot.plot (range (len (sorted_corr [-10:])), sorted_corr[-10:], "*-")

for label, x, y in zip (labels [-10:], range (len (sorted_corr[-10:])), sorted_corr[-10:]):
    matplotlib.pyplot.annotate (label, xy = (x, y), xytext = (-0.5, 0.5),  textcoords='offset points', ha='right', va='bottom')

matplotlib.pyplot.ylabel ("Correlation with the target data")
matplotlib.pyplot.xlabel ("Features")
matplotlib.pyplot.suptitle ("Correlation between features and target")
matplotlib.pyplot.title ("Feature ids are marked above each point")
matplotlib.pyplot.grid ()

# Get the n most correlated features olny
n = 10
n_biggest = sorted_index [-n:]

print (training_feature_set [n_biggest].shape)

lr_results_n_biggest = helper.linearRegression ("data/lr-n-biggest.npy", training_feature_set [n_biggest], training_target_set, \
                                                                         test_feature_set [n_biggest], test_target_set)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_results_n_biggest[1])), lr_results_n_biggest[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_results_n_biggest[2])), lr_results_n_biggest[2], "--", label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle ("N most correlated features linear regression")

matplotlib.pyplot.grid ()

### Normal Form 0 = (XtX)^1X^tY

print ("Normal Form 0 = (XtX)^1X^tY")
norm_results_n_biggest = helper.normalEquation ("data/norm-lr-n-biggest.npy", training_feature_set [n_biggest], training_target_set, \
                                                                              test_feature_set [n_biggest], test_target_set, lr_results_n_biggest[0])

fig, ax = matplotlib.pyplot.subplots (2)

traning_plot, = ax[0].plot (norm_results_n_biggest[1], norm_results_n_biggest[2], label = "Training RMS")
test_plot, = ax[0].plot (norm_results_n_biggest[1], norm_results_n_biggest[3], "--", label = "Testing RMS")

ax[0].set_ylabel ("Root Mean Square Error")
ax[0].set_xlabel ("# training samples")
ax[0].legend (handles = [traning_plot, test_plot])
ax[0].set_title  ("Normal equation RMS errors\nwith the N most correlated features")
ax[0].grid ()

ax[1].plot (norm_results_n_biggest[1], norm_results_n_biggest[4])
ax[1].set_ylabel ("Root Mean Square Error")
ax[1].set_xlabel ("# training samples")
ax[1].set_title (r"RMS between LR and normal equations $\theta$ arrays with N most correlated features")
ax[1].grid ()

matplotlib.pyplot.tight_layout()

### Compare results
fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot_n_biggest, = ax.plot (range(len (lr_results_n_biggest[1])), lr_results_n_biggest[1], label = "Training RMS - N most correlated features")
test_plot_n_biggest, = ax.plot (range(len (lr_results_n_biggest[2])), lr_results_n_biggest[2], "--", label = "Testing RMS - N most correlated features")
traning_plot, = ax.plot (range(len (lr_results[1])), lr_results[1],
"-.", label = "Training RMS")
test_plot, = ax.plot (range(len (lr_results[2])), lr_results[2], label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot_n_biggest, test_plot_n_biggest, traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n$")
matplotlib.pyplot.suptitle ("Linear regression\nComparison between regular and N most correlated features RMS error")

matplotlib.pyplot.grid ()

matplotlib.pyplot.show (block = True)

# More complex models

print ("Linear Regression - Second Order")

training_feature_set_second = numpy.append (training_feature_set, training_feature_set ** 2, axis = 0)
print (training_feature_set_second.shape)

test_feature_set_second = numpy.append (test_feature_set, test_feature_set ** 2, axis = 0)
print (test_feature_set_second.shape)

lr_results_second = helper.linearRegression ("data/lr-second-order.npy", training_feature_set_second, training_target_set, \
                                                                         test_feature_set_second, test_target_set)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_results_second[1])), lr_results_second[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_results_second[2])), lr_results_second[2], "--", label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n + \theta_{n+1}X_1^2 + \theta_{n+2}X_2^2 + ... $")
matplotlib.pyplot.suptitle ("Linear regression with second order features")

matplotlib.pyplot.grid ()

### Normal Form 0 = (XtX)^1X^tY

print ("Normal Form 0 = (XtX)^1X^tY")
norm_results_second = helper.normalEquation ("data/norm-lr-second.npy", training_feature_set_second, training_target_set, test_feature_set_second, test_target_set, lr_results_second[0])

fig, ax = matplotlib.pyplot.subplots (2)

traning_plot, = ax[0].plot (norm_results_second[1], norm_results_second[2], label = "Training RMS")
test_plot, = ax[0].plot (norm_results_second[1], norm_results_second[3], "--", label = "Testing RMS")

ax[0].set_ylabel ("Root Mean Square Error")
ax[0].set_xlabel ("# training samples")
ax[0].legend (handles = [traning_plot, test_plot])
ax[0].set_title  ("Normal equation RMS errors with second order features")
ax[0].grid ()

ax[1].plot (norm_results_second[1], norm_results_second[4])
ax[1].set_ylabel ("Root Mean Square Error")
ax[1].set_xlabel ("# training samples")
ax[1].set_title (r"RMS between LR and normal equations $\theta$ arrays with second order features")
ax[1].grid ()

matplotlib.pyplot.tight_layout()

print ("Minimum RMS with Normal Equation (Second Order) = " + str (min (norm_results_second[3])))
print ("Minimum RMS with LR (Second Order) = " + str (min (lr_results_second[2])))

print ("Linear Regression - Third order")

training_feature_set_third = numpy.append (training_feature_set_second, training_feature_set ** 3, axis = 0)
print (training_feature_set_third.shape)

test_feature_set_third = numpy.append (test_feature_set_second, test_feature_set ** 3, axis = 0)
print (test_feature_set_third.shape)

lr_results_third = helper.linearRegression ("data/lr-third-order.npy", training_feature_set_third, training_target_set, \
                                                                       test_feature_set_third, test_target_set)

fig, ax = matplotlib.pyplot.subplots (1, sharex = True)

traning_plot, = ax.plot (range(len (lr_results_third[1])), lr_results_third[1], label = "Training RMS")
test_plot, = ax.plot (range(len (lr_results_third[2])), lr_results_third[2], "--", label = "Testing RMS")

matplotlib.pyplot.ylabel ("Root Mean Square Error")
matplotlib.pyplot.xlabel ("# iteration")
matplotlib.pyplot.legend(handles = [traning_plot, test_plot])
matplotlib.pyplot.title (r"$Y = \theta_0X_0 + \theta_1X_1 + \theta_2X_2 + ... + \theta_nX_n + \theta_{n+1}X_1^2 + \theta_{n+2}X_2^2 + ... + \theta_{2n+1}X_1^3 + \theta_{2n+2}X_2^3 + ... $")
matplotlib.pyplot.suptitle ("Linear regression RMS errors with third order features")

matplotlib.pyplot.grid ()

print ("Normal Form 0 = (XtX)^1X^tY")
norm_results_third = helper.normalEquation ("data/norm-lr-third.npy", training_feature_set_third, training_target_set, test_feature_set_third, test_target_set, lr_results_third[0])

fig, ax = matplotlib.pyplot.subplots (2)

traning_plot, = ax[0].plot (norm_results_third[1], norm_results_third[2], label = "Training RMS")
test_plot, = ax[0].plot (norm_results_third[1], norm_results_third[3], "--", label = "Testing RMS")

ax[0].set_ylabel ("Root Mean Square Error")
ax[0].set_xlabel ("# training samples")
ax[0].legend (handles = [traning_plot, test_plot])
ax[0].set_title  ("Normal equation RMS erros with third order features")
ax[0].grid ()

ax[1].plot (norm_results_third[1], norm_results_third[4])
ax[1].set_ylabel ("Root Mean Square Error")
ax[1].set_xlabel ("# training samples")
ax[1].set_title (r"RMS between LR and normal equations $\theta$ arrays with third order features")
ax[1].grid ()

matplotlib.pyplot.tight_layout()

print ("Minimum RMS with Normal Equation (Third Order) = " + str (min (norm_results_third[3])))
print ("Minimum RMS with LR (Third Order) = " + str (min (lr_results_third[2])))

matplotlib.pyplot.show (block = True)
