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
        for ind, entry in enumerate(phishingData):
            if ind < maxURLs:
                self.phishingURLs.append(entry['url'])
            else:
                break

        with open(cleanFile) as file:
            cleanReader = csv.reader(file)
            for ind, row in enumerate(cleanReader):
                if ind <= maxURLs:
                    self.cleanURLs.append("https://" + row[0][0:-1])
                else:
                    break
        # remove labels from clean URL list
        self.cleanURLs.pop(0)

    # return list of clean urls
    def getCleanURLs(self, numURLs = 10000):
        numURLs = min(len(self.cleanURLs), numURLs)
        return self.cleanURLs[0:numURLs - 1]

    # return list of phishing urls, will randomly sample the phishing urls
    def getPhishingURLs(self, numURLs = 10000):
        numURLs = min(len(self.phishingURLs), numURLs)
        return random.sample(self.phishingURLs, numURLs)

    phishingURLs = []
    cleanURLs = []

# testing object
if __name__ == '__main__':
    test = Websites()
    phishing = test.getPhishingURLs(numURLs = 100)
    clean = test.getCleanURLs(numURLs = 100)
    print(len(phishing[0]))
    print(len(clean[0]))