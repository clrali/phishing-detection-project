import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from datasetCreation import columns

import pickle

# taken from scikit learn site "comparing classifiers"

names = [
    "Nearest_Neighbors",
    "Linear_SVM",
    "RBF_SVM",
    # "Gaussian_Process",
    "Decision_Tree",
    "Random_Forest",
    "Neural_Net",
    "AdaBoost",
    "Naive_Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42), too big, blows up ram stack
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    AdaBoostClassifier(random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

# read training data
legitimateDF = pd.read_csv("datasets/structured_legitimate_data.csv")
phishingDF = pd.read_csv("datasets/structured_phishing_data.csv")

# combine all data, may want to iterate on this and
# not do this (real world has much less phishing sites compared to benign sites)
totalDF = pd.concat([legitimateDF, phishingDF], axis = 0).sample(frac=1).drop_duplicates()


# implement file based configs on what features we wish to train on

features = columns
commonFeatures = []
try:
    with open(features.json, 'r') as featureFile:
        config = json.load(featureFile)
        for entry in config:
            if entry in features:
                commonFeatures.append(entry)
except Exception as e:
    print("no config file found, using defaults")
# use commonFeatures here to scrub dataset
scrubbedX = totalDF[commonFeatures]
# Dropping URL for now, examine ways to integrate this data for later
X = totalDF.drop(['URL', 'label'], axis = 1)
# accumulate answers to X in Y
Y = totalDF['label']

# default randomize RNG, can set a value if required for repeateable results
# can use this or use kfold validation for training
seed = None
xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size = 0.3, random_state = seed)

for modelName, clf in zip(names, classifiers):
    startTime = time.time()
    print("training classifier: " + modelName)
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(xTrain, yTrain)
    endTime = time.time()
    trainTime = endTime - startTime;
    score = clf.score(xTest, yTest)
    modelFilename = "models/" + modelName + ".pickle"
    resultsFilename = "results/" + modelName + ".txt"
    try:
        with open(modelFilename, 'w+b') as modelFile:
            pickle.dump(clf, modelFile)
    except Exception as e:
        with open(modelFilename, 'xb') as modelFile:
            print(modelFilename + " doesn't exist, creating file")
            pickle.dump(clf, modelFile)
    try:
        with open(resultsFilename, 'w+') as resultsFile:
            resultsFile.write(modelName + " score: " + score + "\n")
            resultsFile.write(modelName + " training time: " + trainTime)
    except Exception as e:
        with open(resultsFilename, 'w+') as resultsFile:
            print(resultsFilename + " doesn't exist, creating file")
            resultsFile.write(f"{modelName} score: {score}\n")
            resultsFile.write(f"{modelName} training time: {trainTime}")
        

# can implement KFold validation quickly if desired, increases runtime
K = 10
kFoldValidationSets  = KFold(n_splits = K, shuffle = True, random_state = seed)

results = {}

""" Using K-fold validation here, implement later
for trainInd, testInd in kFoldValidationSets.split(X):
    Xtrain
"""