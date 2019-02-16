# -*- coding: utf-8 -*-
"""
Created on Sat May  5 12:32:20 2018

@author: Kel3vra
"""

import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn import svm
from sklearn import linear_model
import preprocessing_authors as pr
import vectors as vec
import models as md
import pts as pts

data = pr.getData('train.csv')
print(data.columns.values)



group = pr.groupbyAuthor(data,'author')
print(pr.getUniqueAuthors(data,'author'))
encoded_data = pr.encodeAuthors(data)

#plots
pts.plotInitialData(data)
pts.plotInitialData2(data)

#split our data to (X,y) respectively
x_train, x_test, y_train, y_test = pr.splitData(encoded_data,'text','author')

#Transform the Train set
count_vectorizer,feature_names = vec.countVec()
X_train_idf = vec.tfidfTransform(count_vectorizer)

#Transform the Test set
count_vectorizer_test,feature_names = vec.countVec(x_test)
x_test_idf = vec.tfidfTransform(count_vectorizer_test)

#print our sparse matrix and the word tha we have extracted
print(count_vectorizer_test)
count_vectorizer_test.toarray()
feature_names.get_feature_names()


#predictions with Multinomial Naive Baysen and Linear Svm
predictions_multinomial_Baysen = md.MultinomialNaiveBaysen(X_train_idf, y_train, x_test_idf)
predictions_linearSvm = md.LinearSVM(X_train_idf, y_train, x_test_idf, penality=1)
#models
print("################ Multinomial Naive Baysen ################ ")
# Prediction on train
t = time.time()
MNB_predict_train = md.MultinomialNaiveBaysen(X_train_idf, y_train, x_train, x_train)
print("Accuracy on the train set ",round(accuracy_score(y_train, MNB_predict_train)*100,2),"%")
print(classification_report(y_train, MNB_predict_train))
print("it took", time.time()-t)
# Prediction on the test
t = time.time()
MNB_predict_test = md.MultinomialNaiveBaysen(X_train_idf, y_train, x_train, x_test)
print("Accuracy on the test set ",round(accuracy_score(y_test, MNB_predict_test)*100,2),"%")
print(classification_report(y_test, MNB_predict_test))
print("it took", time.time()-t)

print("################ Linear Support Vector Machine Classifer ################ ")
print("## Penality = 1")
t = time.time()
LinearSVM_predict = md.LinearSVM(X_train_idf, x_train, y_train, x_test)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2))
print(classification_report(y_test, LinearSVM_predict))
print("it took", time.time()-t)


print("################ Linear Support Vector Machine Classifer ################ ")
print("## Penality = 0.5")
t = time.time()
LinearSVM_predict = md.LinearSVM(X_train_idf, x_train, y_train, x_test, 0.5)
print("Accuracy on the test set LINEAR SVM",round(accuracy_score(y_test, LinearSVM_predict)*100,2))
print(classification_report(y_test, LinearSVM_predict))
print("it took", time.time()-t)

print("################ Logistic Regression ################ ")
t = time.time()
model = linear_model.LogisticRegression()
model.fit(X_train_idf, y_train)
LogisticReg_predict = model.predict(tfidf_transformation_test)
print("Accuracy on the test set LOGISTIC REGRESSION",round(accuracy_score(y_test, LogisticReg_predict)*100,2))
print(classification_report(y_test, LogisticReg_predict))
print("it took", time.time()-t)
