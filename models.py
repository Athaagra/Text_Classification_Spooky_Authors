# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:49:00 2018

@author: Kel3vra
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn import svm
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import vectors as vec
def MultinomialNaiveBaysen(X_train_idf, y_train, x_test_idf):
    '''
    params:
    x_train_tfidf: TF-IDF values of the train data
    y_train: Labels of the train data
    data: The x of the data to be predicted
    returns:
    prediction
    '''
    model = MultinomialNB().fit(X_train_idf, y_train)
    prediction = []
    # Predict the authors of the sentences
    for counts in x_test_idf:
        prediction.append(model.predict(counts)[0])
    return prediction

# Linear Suport Vector Machine Classifer
# Penality by default is 1
def LinearSVM(X_train_idf, y_train, x_test_idf, penality=1):
    model = svm.LinearSVC(C=penality)
    model.fit(X_train_idf, y_train)
    prediction=[]
    for sentences in x_test_idf:
        prediction.append(model.predict(sentences)[0])
    return prediction