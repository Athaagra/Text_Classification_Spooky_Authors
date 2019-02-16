# -*- coding: utf-8 -*-
"""
Created on Wed May  9 01:37:28 2018

@author: Kel3vra
"""

from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import prepro_plots as pr
import preprocessing_authors as pra
import vectors as vec
import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import scikitplot as skplt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

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
words = pr.lower_split(words,'word',stop)


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

pd.set_option('display.width', 100)
pd.set_option('precision', 3)
description = data.describe()
print(description)









def plotInitialData(data):
    authors = pr.getUniqueAuthors(data)
    byAuthor = pr.groupbyAuthor(data)
    plt.figure()
    plt.bar(authors, byAuthor.count()['text'])
    plt.show()
data['text'] = data.lines.str.strip('[\W_]+')
data.head()
group = pr.groupbyAuthor(data,'author')
print(pr.getUniqueAuthors(data,'author'))
encoded_data = pr.encodeAuthors(data)


plotInitialData(data)

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
top_n = feature_array[tfidf_sorting][:n]

clf = DecisionTreeClassifier(random_state=42)
clf.fit(x_train_idf,y_train)
pr.print_score(clf, x_train_idf, y_train, x_test_idf, y_test, train=False)


import numpy as np # linear algebra
import pandas as pd 
import matplotlib as mpl






data = data = pr.getData('train.csv')





#Scale Normalization
#sc_x = StandardScaler()
#x_train_idf = sc_x.fit_transform(x_train)

#Classifier
C = 1.0
clf = svm.SVC(kernel='linear', C=C)
clf.fit(x_train_idf, y_train)

#Cross Validation within Train Dataset
res = cross_val_score(clf, x_train_idf, y_train, cv=10, scoring='accuracy')
print('Average Accuracy: \t {0:.4f}'.format(np.mean(res)))
print('Accuracy SD: \t\t {0:.4f}'.format(np.std(res)))

y_train_pred = cross_val_predict(clf, x_train_idf, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)
print("Precision Score: \t {}".format(precision_score(y_train,
                                                          y_train_pred,
                                                          average='weighted')))
print("Recall Score: \t {}".format(recall_score(y_train,
                                                          y_train_pred,
                                                          average='weighted')))
print("F1 Score: \t {}".format(f1_score(y_train,
                                              y_train_pred,
                                                average='weighted')))        
#Cross Validation wirhin Test Dataset                   
                                                          
y_test_pred = cross_val_predict(clf, x_test_idf, y_test, cv=3)
confusion_matrix(y_test, y_test_pred)                         
print("Precision Score: \t {}".format(precision_score(y_test,
                                                          y_test_pred,
                                                          average='weighted')))
print("Recall Score: \t {}".format(recall_score(y_test,
                                                          y_test_pred,
                                                          average='weighted')))
print("F1 Score: \t {}".format(f1_score(y_test,
                                              y_test_pred,
                                                average='weighted')))                                    


skplt.metrics.plot_roc_curve(y_train,y_train_pred)
plt.show()















# classifier
y = label_binarize(y, classes=[0,1,2])
n_classes = 3

#clf = OneVsRestClassifier(linear_model.LogisticRegression())
clf = OneVsRestClassifier(svm.SVC(kernel='linear', C=1.0))
y_score = clf.fit(x_train_idf, y_train).decision_function(x_test_idf)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVC')
    plt.legend(loc="lower right")
    plt.show()


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# For each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# A "micro-average": quantifying score on all classes jointly
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

plt.figure()
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))