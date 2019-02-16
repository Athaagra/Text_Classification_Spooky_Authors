# -*- coding: utf-8 -*-
"""
Created on Sat May  5 12:38:22 2018

@author: Kel3vra
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read the data into a data frame
def getData(path):
    texts = pd.read_csv(path)
    return texts

# Group sentences by the authors
def groupbyAuthor(data,text):
    byAuthor = data.groupby(text)
    return byAuthor

# Encode the authots 'EAP': 0, 'HPL': 1, 'MWS': 2
def encodeAuthors(data):
    data['author'] = data['author'].map({'EAP': 0, 'HPL': 1, 'MWS': 2})
    encoded_data = data
    return encoded_data

# Get the unique authors names as a list
def getUniqueAuthors(data,text):
    authors = list(set(data[text]))
    return authors

# Split the data to a train set and a test set
def splitData(data,x,y):
    x = data[x]
    y = data[y]
    # Splitting data to 70% Train and 30% Test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    return x_train, x_test, y_train, y_test

def print_score(clf, X_train,y_train,X_test, y_test, train=True):
    if train:
        print('Train Result:\n')
        print("accuracy score:{0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {} \n".format(confusion_matrix(y_train, clf.predict(X_train))))
        
        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print('Average Accuracy: \t {0:.4f}'.format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
    elif train==False:
        '''''
        test
        '''''
        print('Test Result: \n')
        print("accuracy score: {0:.4f} \n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {} \n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {} \n".format(confusion_matrix(y_test, clf.predict(X_test))))

