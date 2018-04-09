import matplotlib.pyplot
import numpy
import os
import pywt
import scipy.stats
import skimage.io
import skimage.restoration
import threading

class Classifier ():

    def __init__ (self, trainingAddr = "train/", testAddr = "test/"):

        self.trainingAddr = trainingAddr
        self.testAddr = testAddr

    def extractFeatureThread (self, subdir):

        for file in os.listdir (self.trainingAddr + "/" + subdir)[0:1]:

            print (self.trainingAddr + "/" + subdir + "/" + file)

            noisedImage = skimage.img_as_float(skimage.io.imread (self.trainingAddr + "/" + subdir + "/" + file))

            noiseLoadPath = "%s/%s/fingerprint_%s.jpg" % (self.trainingAddr, subdir, file)
            denoiseLoadPath = "%s/%s/denoised_%s" % (self.trainingAddr, subdir, file)
            
            if os.path.isfile (noiseLoadPath):
                noise = skimage.img_as_float (skimage.io.imread (noiseLoadPath))
            else:

                scales = 4
                features = numpy.array ([])

                # Apply wavelet trasform
                for colorChannel in range (0, 3):
                    # wvt_decomposition = [cAn, (cHn, cVn, cDn), … (cH1, cV1, cD1)]
                    wvt_decomposition = pywt.wavedec2 (data = noisedImage [:,:,colorChannel], wavelet = "db8", level = scales)

                    # scale = (cHn, cVn, cDn), … (cH1, cV1, cD1)
                    for scale in wvt_decomposition [1:]:
                        for direction in range (len (scale)):
                            # Compute mean, variance, skewness and kurtosis
                            mean = numpy.mean (scale [direction])
                            print ("Mean " + str (mean))
                            variation = numpy.var (scale [direction])
                            print ("Variance " + str (variation))
                            skewness = scipy.stats.skew (numpy.histogram (scale [direction])[0])
                            print ("Skewness " + str(skewness))
                            kurtosis = scipy.stats.kurtosis (numpy.histogram (scale [direction])[0])
                            print ("Kurtosis " + str(kurtosis))

                            features = numpy.append (features, [mean, variation, skewness, kurtosis])

                print (features.shape)
                print (features)

                # denoisedImg = skimage.restoration.denoise_wavelet (noisedImg)
                # skimage.io.imsave (denoiseLoadPath, denoisedImg)
                # noise = noisedImg - denoisedImg
                # skimage.io.imsave (noiseLoadPath, noise)

                # print (noisedImg.shape)
                # print (denoisedImg.shape)

    def extractFeatures (self):

        for i, subdir in enumerate (os.listdir (self.trainingAddr)[5:6]):
            t = threading.Thread (target = self.extractFeatureThread, args = (subdir,))
            t.start ()

