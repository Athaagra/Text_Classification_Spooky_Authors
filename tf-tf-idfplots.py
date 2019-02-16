# -*- coding: utf-8 -*-
"""
Created on Thu May 24 21:19:57 2018

@author: Kel3vra
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  9 01:37:28 2018

@author: Kel3vra
"""
import matplotlib.pyplot as plt
import prepro_plots as pr
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import vectors as vec
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
import preprocessing_authors as pra

stop = stopwords.words('english')
stop.append('one')
stop.append('would')
stop.append('could')


data = pr.getData('train.csv')
print(data.columns.values)
pr.plt_authors(data)
#tokenization
token_text = pr.tokenization(data,'id','text')
#from sentences to words
rows = list()
cluster = pr.cluster_words(rows,token_text)

words = pd.DataFrame(cluster, columns=['author', 'word'])
words.head()

#lower , stopwords, removing gaps
words = pr.lowersplit(words,'word',stop)


# =============================================================================
# #top five
# =============================================================================
counts = words.groupby('author')\
    .word.value_counts()\
    .to_frame()\
    .rename(columns={'word':'n_w'})
counts.head()
#plot count frequencies
pr.pretty_plot_top_nw(counts['n_w'])

#number of documents
word_sum = counts.groupby(level=0)\
    .sum()\
    .rename(columns={'n_w': 'n_d'})
word_sum
# tf
tf = counts.join(word_sum)

tf['tf'] = tf.n_w/tf.n_d

tf.head()
#pretty plot tf
pr.pretty_plot_top_ntf(tf['tf'])
#nunique authors
c_d = words.author.nunique()
c_d
# get the number of unique authors that every term appeared in 
idf = words.groupby('word')\
    .author\
    .nunique()\
    .to_frame()\
    .rename(columns={'author':'i_d'})\
    .sort_values('i_d')
idf.head()
#we calculate the idf
idf['idf'] = np.log(c_d/idf.i_d.values)

idf.head()
#we calculate the tf idf
tf_idf = tf.join(idf)

tf_idf.head()

tf_idf['tf_idf'] = tf_idf.tf * tf_idf.idf
tf_idf.head()

pr.pretty_plot_top_n(tf_idf['tf_idf'])
# most important words
r = words[words.word.str.match('^s')]\
    .groupby('word')\
    .count()\
    .rename(columns={'author': 'n'})\
    .nlargest(10, 'n')
r.plot.bar()
r
#pr.wordcloud(data,'text')
data['text'] = data.lines.str.strip('[\W_]+')
data.head()
group = pra.groupbyAuthor(data,'author')
print(pra.getUniqueAuthors(data,'author'))
encoded_data = pra.encodeAuthors(data)
X, y = encoded_data['text'], encoded_data['author']

#y = label_binarize(y, classes=[0,1,2])
#n_classes = 3

# shuffle and split training and test sets
x_train, x_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.20, random_state=42)

#Transform the Train set
count_vectorizer,feature_names = vec.countVec(x_train)
x_train_idf = vec.tfidfTransform(count_vectorizer)

#Transform the Test set
count_vectorizer_test,feature_names = vec.countVec(data['text'])
x_test_idf = vec.tfidfTransform(count_vectorizer_test)


n = 3
#top_n = feature_array[tfidf_sorting][:n]

clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train_idf,y_train)
pra.print_score(clf, x_train_idf, y_train, x_test_idf, y_test, train=False)
