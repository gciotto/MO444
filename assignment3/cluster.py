import csv
import logging
import math
import os
import pickle
import threading

from datetime import datetime
from collections import Counter
from nltk import ngrams

class Cluster ():

    def __init__ (self, csv_file = "news_headlines.csv"):

        logging.basicConfig(filename = 'idfs.log',level=logging.DEBUG)

        self.csvReader = csv.reader (open (csv_file, "r"), delimiter = ",")

        self.wordGramsIDFDict = {}
        self.wordGrams = {}

    def readCsv (self):

        self.dates = []
        self.headlines = []

        for i,row in enumerate(self.csvReader):

            try:
                self.dates.append (datetime.strptime (row [0], "%Y%m%d"))
                self.headlines.append (row [1])
            except:
                print ("Exception at row " + str(i))
        


    def computeIDF (self, element, dictionnary, listElements):

        if element not in dictionnary.keys ():
            counter = 0

            for l in listElements:
                if element in l:
                    counter = counter + 1

            dictionnary [element] = math.log (len(listElements) / counter)
        else:
            print (dictionnary [element])

        return dictionnary [element]

    def computeTermIDFs (self):

        print ("Computing terms")

        if os.path.isfile ("termIDFs.pkl"):
            self.termIDFDict = pickle.load (open("termIDFs.pkl", "rb"))
        else:

            self.termIDFDict = {}

            print (len(self.headlines))

            for i, headline in enumerate(self.headlines):

                # Compute terms IDF
                terms = headline.split()
                for term in terms:
                    self.computeIDF (term, self.termIDFDict, self.headlines)

                logging.info ("Term " + str(i) + "th computed")

            pickle.dump (self.termIDFDict, open("termIDFs.pkl", "wb"))


    def computeChar4GramIDFs (self):

        print ("Computing char 4-grams")

        # Compute character level 4-grams IDF
        if os.path.isfile ("char4gramsIDFs.pkl"):
            self.char4gramsIDFDict = pickle.load (open("char4gramsIDFs.pkl", "rb"))
        else:
            self.char4grams = []

            self.char4gramsIDFDict = {}

            for headline in self.headlines:

                char4grams = []

                # Compute all char level 4-grams
                for gram in ngrams (" " + headline + " ", 4):
                    if gram not in char4grams:
                        char4grams.append (gram)

                self.char4grams.append (char4grams)

            print (len (self.char4grams))

            for i, titleChar4Grams in enumerate(self.char4grams):

                # Compute all char level 4-grams
                for gram in titleChar4Grams:
                    self.computeIDF (gram, self.char4gramsIDFDict, self.char4grams)

                logging.info ("Char 4-gram " + str(i) + "th computed")

            pickle.dump (self.char4gramsIDFDict, open("char4gramsIDFs.pkl", "wb"))    

            print ("tchau")

    def computeWordNGramIDFs (self, n = 3):

        print ("Computing word " + str(n) + " grams")

        # Compute character level 4-grams IDF
        if os.path.isfile ("word" + str(n) + "gramsIDFs.pkl"):
            self.wordGramsIDFDict[n] = pickle.load (open("word" + str(n) + "gramsIDFs.pkl", "rb"))
        else:
            self.wordGrams[n] = []
            self.wordGramsIDFDict[n] = {}

            for headline in self.headlines:

                wordGrams = []

                # Compute all word level n-grams
                for gram in ngrams (("BEG " + headline + " END").split (), n):
                    if gram not in wordGrams:
                        wordGrams.append (gram)

                self.wordGrams[n].append (wordGrams)

            print (len (self.wordGrams[n]))

            for i, titleWordGrams in enumerate(self.wordGrams[n]):

                # Compute all char level 4-grams
                for gram in titleWordGrams:
                    self.computeIDF (gram, self.wordGramsIDFDict[n], self.wordGrams[n])

                logging.info ("Word " + str(n) + "-gram " + str(i) + "th computed") 

            pickle.dump (self.wordGramsIDFDict[n], open("word" + str(n) + "gramsIDFs.pkl", "wb"))

            print ("tchau")

    def preprocess (self):
    
        termThread = threading.Thread (target = self.computeTermIDFs)
        char4gramThread = threading.Thread (target = self.computeChar4GramIDFs)
        wordGramThread = threading.Thread (target = self.computeWordNGramIDFs)

        termThread.start ()
        char4gramThread.start ()
        wordGramThread.start ()

        termThread.join ()
        char4gramThread.join ()
        wordGramThread.join ()

    def computeFeatures (self):

        self.features = []

        for headline in self.headlines:
            # Compute TF-IDF for each term in headline

            terms = headline.split()
            termsSize = len(terms)
            c = Counter (terms)
            termsList = []

            for term in terms:
                termsList.append ((term, c[term] / termsSize * self.computeIDF (term)))

            termsList = sorted(termsList, key=lambda term: -term[1]) 

            # Compute character-level 4-grams

            char4Grams = ngrams (headline, 4)
            termsList = []
        


            
