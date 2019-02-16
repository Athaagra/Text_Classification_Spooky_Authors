# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:42:10 2018

@author: Kel3vra
"""
import matplotlib.pyplot as plt
import preprocessing_authors as pr


# Plot initial data as bar graph
def plotInitialData(data):
    authors = pr.getUniqueAuthors(data)
    byAuthor = pr.groupbyAuthor(data)
    plt.figure()
    plt.bar(authors, byAuthor.count()['text'])
    plt.show()

# Plot initial data as bar graph. (Now it's working)
def plotInitialData2(data):
    data.author.value_counts().plot(kind='bar', rot=0)
    plt.show()