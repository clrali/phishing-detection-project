import json
import csv
import pdb
import random
import asyncio
from sklearn.model_selection import train_test_split

phishingSites = 'phishtank-database/online-valid.json'
cleanSites = 'alexaRank/top1M.csv'

# take in and load data files containing phishing sites and clean sites
class URLLoader:
    def __init__(self, phishingFile = 'phishtank-database/online-valid.json', cleanFile = 'alexaRank/top1M.csv', maxURLs = 100000):
        with open(phishingFile, 'r') as file:
            phishingData = json.load(file)
        for entry in phishingData:
            self.phishingURLs.append(entry['url'])

        with open(cleanFile) as file:
            cleanReader = csv.reader(file)
            for ind, row in enumerate(cleanReader):
                if ind <= maxURLs:
                    self.cleanURLs.append("https://" + row[0][0:-1])
                else:
                    break
        # remove labels from clean URL list
        self.cleanURLs.pop(0)

    # return tuple of two lists, one of length (split * numURLs), and another of length (split * (1-numURLs))
    # will not have duplicates between the two lists, meant for training
    def getCleanURLs(self, numURLs = 10000, split = 0.8):
        numURLs = min(len(self.cleanURLs), numURLs)
        sampleList = random.sample(self.cleanURLs, numURLs)
        firstList = sampleList[0:int(numURLs * split)]
        secondList = sampleList[int(numURLs * split):len(sampleList)]
        return (firstList, secondList)

    # return tuple of two lists, one of length (split * numURLs), and another of length (split * (1-numURLs))
    # will not have duplicates between the two lists, meant for training
    def getPhishingURLs(self, numURLs = 10000, split = 0.8):
        numURLs = min(len(self.phishingURLs), numURLs)
        sampleList = random.sample(self.phishingURLs, numURLs)
        firstList = sampleList[0:int(numURLs * split)]
        secondList = sampleList[int(numURLs * split):len(sampleList)]
        return (firstList, secondList)

    phishingURLs = []
    cleanURLs = []

# testing object
if __name__ == '__main__':
    test = Websites()
    phishing = test.getPhishingURLs(numURLs = 100)
    clean = test.getCleanURLs(numURLs = 100)
    print(len(phishing[0]))
    print(len(clean[0]))