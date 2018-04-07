import cv2
import matplotlib.pyplot
import os
import threading

class Classifier ():

    def __init__ (self, trainingAddr = "train/", testAddr = "test/"):

        self.trainingAddr = trainingAddr
        self.testAddr = testAddr

    def extractFeatureThread (self, subdir):

        for file in os.listdir (self.trainingAddr + "/" + subdir)[0:1]:

            print (self.trainingAddr + "/" + subdir + "/" + file)

            h = 5
            templateWindowSize = 7
            searchWindowSize = 21

            noisedImg = cv2.imread (self.trainingAddr + "/" + subdir + "/" + file)

            denoisedPath = "%s/%s/%s_%i_%i_%i_%i" % (self.trainingAddr, subdir, file, h, h, templateWindowSize, searchWindowSize)

            if os.path.isfile (denoisedPath):
                denoisedImg = cv2.imread (denoisedPath)
            else:
                denoisedImg = cv2.fastNlMeansDenoisingColored (noisedImg, None, h, h, templateWindowSize , searchWindowSize)
                cv2.imwrite (denoisedPath, denoisedImg)

            noise = noisedImg - denoisedImg

    def extractFeatures (self):

        for subdir in os.listdir (self.trainingAddr):
            t = threading.Thread (target = self.extractFeatureThread, args = (subdir,))
            t.start ()
