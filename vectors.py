# -*- coding: utf-8 -*-
"""
Created on Sat May  5 14:07:09 2018

@author: Kel3vra
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk



analyzer = CountVectorizer().build_analyzer()
stemmer = nltk.stem.PorterStemmer()


def stemWords(text):
    return (stemmer.stem(w) for w in analyzer(text))

# Count vectorizer for the train data
def countVec(x_train):
    count_vect = CountVectorizer(stop_words='english', 
                                 token_pattern="\w*[a-z]\w*", 
                                 max_features=2000,
                                 analyzer=stemWords)
    x_train_counts = count_vect.fit_transform(x_train)
    
    return x_train_counts,count_vect

# TF-IDF values based on the count vectorizer
def tfidfTransform(x_train_counts):
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    return x_train_tfidf

# Transform the data to be predicted
def countvecTransform(train_data, data):
    count_vect = CountVectorizer()
    # You need to fit and then transform the test data again
    # To transform the data to be predicted in the same way
    fit = count_vect.fit_transform(train_data)
    transform = []
    # Transform sentence by count vec
    for sentences in data:
        transform.append(count_vect.transform([sentences]))
    return transform