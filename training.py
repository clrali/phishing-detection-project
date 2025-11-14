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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import pickle
from os import listdir

# default randomize RNG, can set a value if required for repeateable results
# can use this or use kfold validation for training
seed = None

# taken from scikit learn site "comparing classifiers"
names = [
    "Logistic_Regression",
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
    LogisticRegression(),
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
columns = [
    'has_title',
    'has_input',
    'has_button',
    'has_image',
    'has_submit',
    'has_link',
    'has_password',
    'has_email_input',
    'has_hidden_element',
    'has_audio',
    'has_video',
    'number_of_inputs',
    'number_of_buttons',
    'number_of_images',
    'number_of_option',
    'number_of_list',
    'number_of_th',
    'number_of_tr',
    'number_of_href',
    'number_of_paragraph',
    'number_of_script',
    'length_of_title',
    'has_h1',
    'has_h2',
    'has_h3',
    'length_of_text',
    'number_of_clickable_button',
    'number_of_a',
    'number_of_img',
    'number_of_div',
    'number_of_figure',
    'has_footer',
    'has_form',
    'has_text_area',
    'has_iframe',
    'has_text_input',
    'number_of_meta',
    'has_nav',
    'has_object',
    'has_picture',
    'number_of_sources',
    'number_of_span',
    'number_of_table',
    'URL'
]
configs = []
try:
    testFiles = listdir('./settings')
    testFiles = [fileName for fileName in testFiles if '.json' in fileName]
    for featureFile in testFiles:
        with open("./settings/" + featureFile, 'r') as settings:
            config = json.load(settings)
            requestedFeatures = config['features']
            requestedFeatures = [setting for setting in requestedFeatures if setting in columns]
            # assume URL and label are there in settings
            if 'URL' not in requestedFeatures:
                requestedFeatures.append('URL')
            if 'label' not in requestedFeatures:
                requestedFeatures.append('label')
            suffix = config['suffix']
            configs.append((requestedFeatures,suffix))
except Exception as e:
    print("no config or invalid file found, using defaults")
    configs.clear()
    configs.append((columns, "default"))

for commonFeatures, fileSuffix in configs:
    # use commonFeatures here to scrub dataset
    print('using common features:')
    print(commonFeatures)
    print("with suffix:  " + fileSuffix)
    scrubbedX = totalDF[commonFeatures]
    # Dropping URL for now, examine ways to integrate this data for later
    X = totalDF.drop(['URL', 'label'], axis = 1)
    # accumulate answers to X in Y
    Y = totalDF['label']
    
    xTrain, xTest, yTrain, yTest = train_test_split(X,Y, test_size = 0.3, random_state = seed)
    
    for modelName, clf in zip(names, classifiers):
        startTime = time.time()
        print("training classifier: " + modelName)
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(xTrain, yTrain)
        endTime = time.time()
        trainTime = endTime - startTime;
        answers = clf.predict(xTest)
        # calculate total score of classifier
        score = sum(answers == yTest) / answers.size
        # extracting values we may find useful
        clf_confusion_matrix = confusion_matrix(yTest, answers)
        tn, fp, fn, tp = clf_confusion_matrix.ravel().tolist()
        modelFilename = "models/" + modelName + fileSuffix + ".pickle"
        resultsFilename = "results/" + modelName + fileSuffix + ".txt"
        # save classifiers for later 
        try:
            with open(modelFilename, 'w+b') as modelFile:
                pickle.dump(clf, modelFile)
        except Exception as e:
            with open(modelFilename, 'xb') as modelFile:
                print(modelFilename + " doesn't exist, creating file")
                pickle.dump(clf, modelFile)
        # save score and metrics for later
        try:
            with open(resultsFilename, 'w+') as resultsFile:
                resultsFile.write(f"{modelName} score: {score}\n")
                resultsFile.write(f"{modelName} True Negative: {tn}, False Positive: {fp}, False Negative: {fn}, True Positive: {tp}\n")
                resultsFile.write(f"{modelName} training time: {trainTime}")
        except Exception as e:
            with open(resultsFilename, 'x+') as resultsFile:
                print(resultsFilename + " doesn't exist, creating file")
                # calculate and add other metrics we would like to present here
                resultsFile.write(f"{modelName} score: {score}\n")
                resultsFile.write(f"{modelName} True Negative: {tn}, False Positive: {fp}, False Negative: {fn}, True Positive: {tp}\n")
                resultsFile.write(f"{modelName} training time: {trainTime}")

# can implement KFold validation quickly if desired, increases runtime
K = 5
kFoldValidationSets  = KFold(n_splits = K, shuffle = True, random_state = seed)

results = {}

""" Using K-fold validation here, implement later
for trainInd, testInd in kFoldValidationSets.split(X):
    Xtrain
"""