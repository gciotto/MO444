import numpy
import math

def normalize (feature_matrix = [], means = None, stds = None):

    mean_feat = []
    std_feat = []

    if len (feature_matrix.shape) == 1:

        if means == None or stds == None:
            mean = numpy.mean (feature_matrix)
            mean_feat.append (mean)
            std = numpy.std (feature_matrix)
            std_feat.append (std)
            return ((feature_matrix - mean) / std, mean_feat, std_feat)

        else:
            mean = means [0]
            std = stds [0]
            return ((feature_matrix - mean) / std, means, stds)

    return_matrix = feature_matrix.copy()

    for f, feature in enumerate (return_matrix [:]):

        if means == None or stds == None:
            mean = numpy.mean (feature)
            mean_feat.append (mean)
            std = numpy.std (feature)
            std_feat.append (std)
        else:
            mean = means [f]
            std = stds [f]

        feature = (feature - mean) / std
        return_matrix [f] = feature

    if means == None or stds == None:
        return (return_matrix, mean_feat, std_feat)

    return (return_matrix, means, stds)

def appendBias (feature_matrix = []):

    feature_bias = numpy.ones ((1, feature_matrix.shape [1]))
    return numpy.append (feature_bias, feature_matrix, axis = 0)

def evaluateTheta (test_feature_set, test_target_set, theta):

    predicted_target = theta.transpose().dot (test_feature_set)
    return math.sqrt(((test_target_set - predicted_target[0]) ** 2).mean ())

def gradientDescent (training_feature_set, training_target_set, test_feature_set, test_target_set, alpha = 0.1, regularization = 0, max_steps = 30):

    feature_count = training_feature_set.shape [0]
    sample_count = training_feature_set.shape [1]

    thetas = numpy.random.randn (feature_count, 1)

    step = 0

    train_error = []
    test_error = []

    predicted_target = thetas.transpose().dot (training_feature_set)
    rms = math.sqrt(((training_target_set - predicted_target[0]) ** 2).mean ())
    print (rms)

    train_error.append (rms)

    test_error.append (evaluateTheta (test_feature_set, test_target_set, thetas))

    while step < max_steps:

        for j in range (feature_count):

            summ = 0.0

            for i in range (sample_count):
                summ += ((predicted_target [0][i] - training_target_set [i]) * training_feature_set [j][i])

            thetas [j] = thetas[j] - (alpha / sample_count) * (summ + regularization * thetas[j])

        predicted_target = thetas.transpose().dot (training_feature_set)

        rms = math.sqrt (((training_target_set - predicted_target [0]) ** 2).mean ())
        print (rms)

        train_error.append (rms)
        test_error.append (evaluateTheta (test_feature_set, test_target_set, thetas))

        step += 1

        if len (train_error) > 1:
            err = train_error [len (train_error) - 2] - train_error [len (train_error) - 1]

            if err < 0:
                alpha = alpha / 2
                print ("alpha = " + str (alpha))

    return thetas, train_error, test_error

def linearRegression (filename, training_feature_set, training_target_set, test_feature_set, test_target_set, regularization = 0, steps = 100):

    try:
        results = numpy.load (filename)

    except FileNotFoundError:

        # Normalize columns to obtain better and faster results - use (X - u) / sigma, where u = mean and sigma = std. dev.
        lr_training_feature_set, lr_feat_means, lr_feat_stds = normalize (training_feature_set)
        lr_training_target_set, lr_target_means, lr_target_stds = normalize (training_target_set)
       
        lr_test_feature_set, lr_test_feat_means, lr_test_feat_stds = normalize (test_feature_set, means = lr_feat_means, stds = lr_feat_stds)
        lr_test_target_set, lr_test_target_means, lr_test_target_stds = normalize (test_target_set, means = lr_target_means, stds = lr_target_stds)

        # Append bias columns in the features data sets

        lr_training_feature_set = appendBias (lr_training_feature_set)
        lr_test_feature_set = appendBias (lr_test_feature_set)

        # First version of the LR - simple model
        theta_lr, train_error, test_error = gradientDescent (lr_training_feature_set, lr_training_target_set, \
                                                             lr_test_feature_set, lr_test_target_set, \
                                                             alpha = 0.25, regularization = regularization, \
                                                             max_steps = steps)

        results = [theta_lr, train_error, test_error]
        numpy.save (filename, results)

    return results

def normalEquation (filename, training_feature_set, training_target_set, test_feature_set, test_target_set, lr_theta, regularization = 0):

    try:
        norm_results = numpy.load (filename)

    except:

        norm_thetas = []
        norm_sizes = []
        norm_train_errors = []
        norm_test_errors = []
        norm_theta_errors = []

        for feat_size in range (500, training_feature_set.shape [1], 1000):

            # Choose a subset of the features
            indexes = range (feat_size)

            print (training_feature_set [:, indexes].shape)

            norm_training_feature_set, norm_feat_means, norm_feat_stds = normalize (training_feature_set [:, indexes])
            norm_training_target_set, norm_target_means, norm_target_stds = normalize (training_target_set [indexes])

            norm_test_feature_set, norm_test_feat_means, norm_test_feat_stds = normalize (test_feature_set, means = norm_feat_means, stds = norm_feat_stds)
            norm_test_target_set, norm_test_target_means, norm_test_target_stds = normalize (test_target_set, means = norm_target_means, stds = norm_target_stds)

            norm_training_feature_set = appendBias (norm_training_feature_set)
            norm_test_feature_set = appendBias (norm_test_feature_set)

            inv = numpy.linalg.pinv (norm_training_feature_set.dot (norm_training_feature_set.transpose ()) + regularization * numpy.eye (norm_training_feature_set.shape [0]))
            stp = inv.dot (norm_training_feature_set)
            theta_norm = stp.dot (norm_training_target_set)

            norm_thetas.append (theta_norm)
            norm_sizes.append (feat_size)
            norm_train_errors.append (evaluateTheta (norm_training_feature_set, norm_training_target_set, theta_norm))
            norm_test_errors.append (evaluateTheta (norm_test_feature_set, norm_test_target_set, theta_norm))
            norm_theta_errors.append (math.sqrt(((theta_norm - lr_theta) ** 2).mean ())) 

            print ("RMS between LR and norm. = " + str (math.sqrt(((theta_norm - lr_theta) ** 2).mean ())))
            print ("Training RMS = " + str (evaluateTheta (norm_training_feature_set, norm_training_target_set, theta_norm)))
            print ("Testing RMS = " + str (evaluateTheta (norm_test_feature_set, norm_test_target_set, theta_norm)))

        norm_results = [norm_thetas, norm_sizes, norm_train_errors, norm_test_errors, norm_theta_errors]
        numpy.save (filename, norm_results)

    return norm_results
