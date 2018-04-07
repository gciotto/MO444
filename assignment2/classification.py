import cv2
import matplotlib.pyplot
import os
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

            h = 3
            templateWindowSize = 7
            searchWindowSize = 21

            w = 0.5

            noisedImg = skimage.img_as_float(skimage.io.imread (self.trainingAddr + "/" + subdir + "/" + file))
            print (noisedImg)
            # noiseLoadPath = "%s/%s/%s_noise_%i_%i_%i_%i" % (self.trainingAddr, subdir, file, h, h, templateWindowSize, searchWindowSize)
            noiseLoadPath = "%s/%s/%s_noise_wavelet.jpg" % (self.trainingAddr, subdir, file)
            denoiseLoadPath = "%s/%s/denoised_%s" % (self.trainingAddr, subdir, file)
            

            if os.path.isfile (noiseLoadPath):
                noise = cv2.imread (noiseLoadPath)
            else:
                # denoisedImg = cv2.fastNlMeansDenoisingColored (noisedImg, None, h, h, templateWindowSize , searchWindowSize)
                denoisedImg = skimage.restoration.denoise_wavelet (noisedImg)
                skimage.io.imsave (denoiseLoadPath, denoisedImg)
                noise = noisedImg - denoisedImg
                # skimage.io.imsave (noiseLoadPath, noise)

                print (noisedImg.shape)
                print (denoisedImg.shape)



    def extractFeatures (self):

        for i, subdir in enumerate (os.listdir (self.trainingAddr)[5:6]):
            t = threading.Thread (target = self.extractFeatureThread, args = (subdir,))
            t.start ()

