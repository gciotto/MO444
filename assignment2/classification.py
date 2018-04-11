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

class Classifier ():

    def __init__ (self, trainingAddr = "train/", testAddr = "test/"):

        self.trainingAddr = trainingAddr
        self.testAddr = testAddr

    def extractFeatureThread (self, subdir):

        for file in os.listdir (self.trainingAddr + "/" + subdir)[0:10]:

            print (self.trainingAddr + "/" + subdir + "/" + file)

            noisedImageInt = skimage.io.imread (self.trainingAddr + "/" + subdir + "/" + file)
            noisedImage = skimage.img_as_float(noisedImageInt)

            noiseLoadPath = "%s/%s/fingerprint_%s" % (self.trainingAddr, subdir, file)
            denoiseLoadPath = "%s/%s/denoised_%s" % (self.trainingAddr, subdir, file)
            featureLoadPath = "%s/%s/%s" % ("features", subdir, file)

            if not os.path.isfile (featureLoadPath):

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

                        # print (len(filters))
                        denoised_wvt.append (filters)

                    # print (len(denoised_wvt))
                    denoisedImage [:,:, colorChannel] = pywt.waverec2 (denoised_wvt, wavelet = "db8")

                denoisedImage = numpy.maximum (-1, denoisedImage)
                denoisedImage = numpy.minimum (1, denoisedImage)
                noise = noisedImage - denoisedImage
                # Zero-meaning
                noise = (noise - noise.mean ()) / noise.std ()
                noise [:,:, 0] = 0.3 * noise [:,:, 0]
                noise [:,:, 1] = 0.6 * noise [:,:, 1]
                noise [:,:, 2] = 0.1 * noise [:,:, 2]
                noise = numpy.maximum (-1, noise)
                noise = numpy.minimum (1, noise)
                # skimage.io.imsave (noiseLoadPath, noise)
                # skimage.io.imsave (denoiseLoadPath, denoisedImage)

                # print (features.shape)

                # Last features
                for colorChannel in range (0, 3):
                    # wvt_decomposition = [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)]
                    wvt_noise_decomposition = pywt.wavedec2 (data = noise [:,:,colorChannel], wavelet = "db8", level = 1)
                    for scale in wvt_noise_decomposition [1:]:
                        for direction in range (len (scale)):
                            for n in range (1, 10):
                                features = numpy.append (features, [scipy.stats.moment (scale [direction], moment = n, axis = None)])

                    # Co-ocurrence matrix of fingerprint image
                    glcm = skimage.feature.greycomatrix(skimage.img_as_ubyte(noise[:,:,colorChannel]), [1], [0, numpy.pi/2, numpy.pi, 3*numpy.pi/2], normed=True)
                    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
                        p = skimage.feature.greycoprops (glcm, prop = prop)
                        features = numpy.append (features, p[0])

                numpy.save (featureLoadPath, features)

    def extractFeatures (self):

        print (os.listdir (self.trainingAddr))
        for i, subdir in enumerate (os.listdir (self.trainingAddr)[1:3]):
            t = threading.Thread (target = self.extractFeatureThread, args = (subdir,))
            t.start ()
