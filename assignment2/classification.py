import logging
import matplotlib.pyplot
import numpy
import os
import pywt
import scipy.ndimage
import scipy.signal
import scipy.stats
import skimage.feature
import skimage.io
import skimage.restoration
import threading
import cv2
import sklearn.linear_model
import sklearn.metrics
import sklearn.neural_network
import sklearn.externals.joblib
import random
import matplotlib.pyplot as plt
import itertools
import pickle

import csv

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler


class Classifier ():

    def __init__ (self, trainingAddr = "train/", testAddr = "test/"):

        logging.basicConfig(filename = 'feaures.log',level=logging.DEBUG)
        self.trainingAddr = trainingAddr
        self.testAddr = testAddr

        self.csv_lr = open("test.csv", 'wt')
        self.csv_nn = open("test_nn.csv", 'wt')
        self.csvFile = csv.writer(self.csv_lr)
        self.csvFileNN = csv.writer(self.csv_nn)
        self.csvFile.writerow(["fname","camera"])
        self.csvFileNN.writerow(["fname","camera"])

    def extractFeatureThread (self, subdir, featuresPath, model):

        for file in os.listdir (subdir):

            try:
                logging.info (subdir + "/" + file)

                featureLoadPath = "%s/%s/%s" % (featuresPath, model, file)

                if not os.path.isfile (featureLoadPath + ".npy"):

                    noisedImage = cv2.imread (subdir + "/" + file)

                    print (noisedImage.shape)

                    # noisedImage = skimage.img_as_float(noisedImageInt)

                    scales = 4
                    features = numpy.array ([])

                    print (noisedImage.shape)
                    denoisedImage = numpy.zeros (noisedImage.shape)

                    # Apply wavelet trasform
                    for colorChannel in range (0, 3):
                        # wvt_decomposition = [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)]
                        wvt_decomposition = pywt.wavedec2 (data = noisedImage [:,:,colorChannel], wavelet = "db8", level = scales)

                        # scale = (cHn, cVn, cDn), … (cH1, cV1, cD1)
                        denoised_wvt = [wvt_decomposition[0]]
                        for scale in wvt_decomposition [1:]:
                            filters = ()
                            # print (len (scale))
                            for direction in range (len (scale)):
                                # Compute mean, variance, skewness and kurtosis - Higher-order wavelet features - Source Camera Identification Forensics Based on Wavelet Features
                                mean = numpy.mean (scale [direction])
                                # print ("Mean " + str (mean))
                                variation = numpy.var (scale [direction])
                                # print ("Variance " + str (variation))
                                skewness = scipy.stats.skew (numpy.histogram (scale [direction])[0])
                                # print ("Skewness " + str(skewness))
                                kurtosis = scipy.stats.kurtosis (numpy.histogram (scale [direction])[0])
                                # print ("Kurtosis " + str(kurtosis))

                                features = numpy.append (features, [mean, variation, skewness, kurtosis])

                                # Extract PRNU - Source Smartphone Identification Using Sensor Pattern Noise and Wavelet Transform
                                ## Compute local Variance
                                win_rows = 5
                                win_cols = 5
                                win_mean = scipy.ndimage.uniform_filter(scale [direction], (win_rows, win_cols))
                                win_sqr_mean = scipy.ndimage.uniform_filter(scale [direction]**2, (win_rows, win_cols))
                                variance = win_sqr_mean - win_mean**2
                                sigma = 0.001
                                wiener = scale [direction] * (variance**2)/(variance**2 + sigma**2)
                                filters = filters + (wiener,)


                                # Co-ocurrence matrix of fingerprint image
                                glcm = skimage.feature.greycomatrix(scale [direction].astype (numpy.uint8), [5], [0, numpy.pi/2, numpy.pi, 3*numpy.pi/2], normed=True)
                                for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:                           
                                    p = skimage.feature.greycoprops (glcm, prop = prop)
                                    # print (p[0])
                                    for pr in p [0]:
                                        features = numpy.append (features, pr)

                            # print (len(filters))
                            denoised_wvt.append (filters)

                        # print (len(denoised_wvt))
                        denoisedImage [:,:, colorChannel] = pywt.waverec2 (denoised_wvt, wavelet = "db8")

                    # denoisedImage = numpy.maximum (0, denoisedImage)
                    # denoisedImage = numpy.minimum (255, denoisedImage)
                    noise = noisedImage - denoisedImage
                    noise = numpy.maximum (0, noise)
                    noise = numpy.minimum (255, noise)
                    # Zero-meaning
                    noiseZero = (noise - noise.mean ()) / noise.std ()
                    noiseZero [:,:, 0] = 0.3 * noiseZero [:,:, 0]
                    noiseZero [:,:, 1] = 0.6 * noiseZero [:,:, 1]
                    noiseZero [:,:, 2] = 0.1 * noiseZero [:,:, 2]
                    # print (noise)
                    # skimage.io.imsave (noiseLoadPath, noise)
                    # skimage.io.imsave (denoiseLoadPath, denoisedImage)
                    # print (denoisedImage)

                    # Last features
                    for colorChannel in range (0, 3):
                        # wvt_decomposition = [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)]
                        wvt_noise_decomposition = pywt.wavedec2 (data = noise [:,:,colorChannel], wavelet = "db8", level = 4)
                        for scale in wvt_noise_decomposition [1:]:
                            for direction in range (len (scale)):
                                for n in range (1, 10):
                                    features = numpy.append (features, [scipy.stats.moment (scale [direction], moment = n, axis = None)])

                    numpy.save (featureLoadPath, features)

            except Exception as e:
                logging.error (file + " failed with '" + str(e) + "'")

    def extractFeatures (self):

        print (os.listdir (self.trainingAddr))
        threads = []
        for i, subdir in enumerate (os.listdir (self.trainingAddr)):
            if not os.path.exists("features/" + subdir):
                os.makedirs("features/" + subdir)

            t = threading.Thread (target = self.extractFeatureThread, args = (self.trainingAddr + "/" + subdir, "features", subdir))
            threads.append (t)
            t.start ()

        t = threading.Thread (target = self.extractFeatureThread, args = (self.testAddr, "features_test", ""))
        threads.append (t)
        t.start ()

        for t in threads:
            t.join ()

    def fit (self):

        X = []
        classes = []
        target = numpy.array ([])
        targetClasses = numpy.array ([])
        targetArrays = []
        for cl, subdir in enumerate (os.listdir ("features/")):
            classes.append (subdir)
            for file in os.listdir ("features/" + subdir):
                feature = numpy.load ("features/" + subdir + "/" + file)
                X.append(feature)
                target = numpy.append (target, cl)
                targetClasses = numpy.append (targetClasses, subdir)
                tArray = numpy.zeros (10)
                tArray [cl] = 1.0
                targetArrays.append (tArray)


        X = numpy.vstack(X)
        targetArrays = numpy.vstack(targetArrays)

        X_test = []
        fileNames = []
        for file in os.listdir ("features_test/"):
            feature = numpy.load ("features_test/" + file)
            X_test.append(feature)
            fileNames.append (file [:-4])

        X_test = numpy.vstack(X_test)

        print (len(classes))
        print (X.shape)
        print ("Test = " + str(X_test.shape))

        shuffled = numpy.array (range (X.shape [0]))
        numpy.random.shuffle (shuffled)

        print (shuffled)

        X = X [shuffled,:]
        target = target [shuffled]
        targetClasses = targetClasses [shuffled]
        targetArrays = targetArrays [shuffled, :]

        print (X.shape)
        print (target.shape)

        validation = random.sample (range (0, X.shape [0]), int (X.shape [0] * 0.2))
        V = X [validation, :]
        targetV = target [validation]
        targetClassesV = targetClasses [validation]
        targetArraysV = targetArrays [validation, :]

        X = numpy.delete (X, validation, axis = 0)
        target = numpy.delete (target, validation, axis = 0)
        targetClasses = numpy.delete (targetClasses, validation, axis = 0)
        targetArrays = numpy.delete (targetArrays, validation, axis = 0)

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_norm = scaler.fit_transform (X)
        V_norm = scaler.fit_transform (V)
        X_test_norm = scaler.fit_transform (X_test)

        print (V.shape)
        print (targetV.shape)

        print ("X.shape = " + str(X.shape))
        print (target.shape)

        if not os.path.isfile ("model/model_logistic.pkl"):

            # Logistic Regression
            regularization = 100.0
            model = sklearn.linear_model.LogisticRegression (C = 1/regularization, multi_class = "ovr", max_iter = 100, n_jobs = 4)
            model = model.fit (X, target)

            sklearn.externals.joblib.dump (model, "model/model_logistic.pkl")

        else:
            model = sklearn.externals.joblib.load ("model/model_logistic.pkl")

        labels = model.predict (V)
        labels_test_lr = model.predict (X_test)
        print ("merdaa " + str(labels_test_lr.shape))

        for i, cl in enumerate (labels_test_lr):
            self.csvFile.writerow([fileNames [i], classes [cl.astype(int)]])

        self.csv_lr.flush()

        print (i)
        print (labels_test_lr)

        # Compute confusion matrix
        cnf_matrix = sklearn.metrics.confusion_matrix (targetV, labels)
        numpy.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classes,
                              title='Confusion matrix, without normalization - Logistic Regression')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                                   title='Normalized confusion matrix - Logistic Regression', fileName = "confusion_matrix/logistic.png")

        earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

        if not os.path.isfile ("model/model_nn.hd5"):

            print (X.shape [1])
            # Neural Networks
            # nn = sklearn.neural_network.MLPClassifier (hidden_layer_sizes = (X.shape [1]), activation = "logistic", max_iter = 1000)
            # nn = nn.fit (X, target)            
            # print (nn.classes_)

            nn = Sequential()

            nn.add(Dense(units = X.shape [1], activation = 'sigmoid', input_dim = X.shape [1]))
            nn.add(Dense(units  = 10, activation='sigmoid'))
            nn.compile(optimizer='adam', loss='mean_squared_error', metrics = ['accuracy'])           

            for i in range (100):
                print (i)
                nn.fit(X_norm, targetArrays, epochs = 10, batch_size = 16, validation_data = (V_norm, targetArraysV), callbacks = [earlyStopping])

            sklearn.externals.joblib.dump (scaler, "model/scaler.pkl")
            nn.save ("model/model_nn.hd5")
            #sklearn.externals.joblib.dump (nn, "model/model_nn.pkl")

        else:
            #nn = sklearn.externals.joblib.load ("model/model_nn.pkl")
            nn = load_model("model/model_nn.hd5")
            scaler = sklearn.externals.joblib.load ("model/scaler.pkl")

        # labels = nn.predict (V)
        labels_nn = nn.predict(V_norm, batch_size = 16)
        labels_test_nn = nn.predict(X_test_norm, batch_size = 16)

        classes_nn = []
        for label in labels_nn:   
            classes_nn.append (numpy.argmax(label))

        classes_test_nn = []
        for label in labels_test_nn:   
            classes_test_nn.append (numpy.argmax(label))

        for i, cl in enumerate (classes_test_nn):
            self.csvFileNN.writerow([fileNames [i], classes [cl.astype(int)]])

        self.csv_nn.flush()

        numpy.set_printoptions(threshold=numpy.nan)
        print (targetArrays)
        print (labels)
        print (targetV)

        # Compute confusion matrix
        cnf_matrix = sklearn.metrics.confusion_matrix (targetV, classes_nn)
        numpy.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classes,
                              title='Confusion matrix, without normalization - Neural Networks')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                                   title='Normalized confusion matrix - Neural Networks') 


        # plt.show (block = True)

        print (sklearn.metrics.mean_absolute_error(classes_test_nn, labels_test_lr))

        print ((classes_test_nn == labels_test_lr).sum ()/ len (labels_test_lr))

        labels_nn = []
        for i in range (10):

            if not os.path.isfile ("model/model_nn_" + str(i) + ".hd5"):
                nn = Sequential()

                nn.add(Dense(units = X.shape [1], activation = 'sigmoid', input_dim = X.shape [1]))
                nn.add(Dense(units  = 1, activation='sigmoid'))
                nn.compile(optimizer = 'adam', loss='mean_squared_error', metrics = ['binary_accuracy'])           

                for j in range (100):
                    print (j)
                    nn.fit(X_norm, targetArrays [:, i], epochs = 10, batch_size = 16,  validation_data = (V_norm, targetArraysV [:, i]), callbacks = [earlyStopping])

                nn.save ("model/model_nn_" + str(i) + ".hd5")

            else:

                nn = load_model("model/model_nn_" + str(i) + ".hd5")

            l = nn.predict(V_norm, batch_size = 16)
            labels_nn.append (l)

        print (len(labels_nn))
        labels_nn = numpy.array (labels_nn)
        print (labels_nn.shape)
        classes_nn = []
        for sample in range (V_norm.shape [0]):   
            classes_nn.append(numpy.argmax(labels_nn[:, sample]))

        cnf_matrix = sklearn.metrics.confusion_matrix (targetV, classes_nn)

        numpy.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classes,
                              title='Confusion matrix, without normalization - Neural Networks')

        # Plot normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                                   title='Normalized confusion matrix - Neural Networks') 

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, fileName = ""):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = numpy.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

