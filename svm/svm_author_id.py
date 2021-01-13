#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# alter C value and kernel type to optimize SVM for the data set, this is pretty much optimal for this set
clf = SVC(C=10000.0, kernel="rbf")
print "clf created..."

# use to shrink data set to reduce traininf & prediction time (for testing purposes)
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

training_start = time()
print "training clf..."
clf.fit(features_train, labels_train)
training_end = time()

print "Training time: {}".format(training_end - training_start)

print "predicting..."

predict_start = time()
pred = clf.predict(features_test)
predict_end = time()

print "Prediction time: {}".format(predict_end - predict_start)

from sklearn.metrics import accuracy_score
print "Accuracy: {}".format(accuracy_score(pred, labels_test))


# Determine how many emails were predicted to be sent by Chris
count = 0

for res in pred:
    if res == 1:
        count = count+1

print "Total: {}, Chris: {}".format(len(pred), count)
