import os
import cv2
import matplotlib.pyplot

class Classifier ():

    def __init__ (self, trainingAddr = "train/", testAddr = "test/"):

        self.trainingAddr = trainingAddr
        self.testAddr = testAddr

    def extractFeatures (self):

        for subdir in os.listdir (self.trainingAddr):
            for file in os.listdir (self.trainingAddr + "/" + subdir):

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
