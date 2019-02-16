# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:27:20 2018

@author: Kel3vra
"""

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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib as mpl
#wordcloud = WordCloud(width=800, height=400).generate(text)
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['font.size'] = 14
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1
# Read the data into a data frame
def getData(path):
    texts = pd.read_csv(path)
    return texts

def plt_authors(data):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    a = data.groupby('author').count()
    a = a.drop(['text'], axis=1)
    #ax.legend(['authors'])
    plt.figure
    plt.figure(figsize=(16,8))
    # plot chart
    #ax1 = plt.subplot(121, aspect='equal')
    a.plot.bar(subplots=True,figsize=(8, 6))
    #a.legend(['authors'])
    plt.savefig('spooky.png')
    plt.show()
   

def tokenization(data,drop_column,text):
    data_cl = data.drop([drop_column], axis=1)
    data_cl[text] = data_cl.text.str.strip().str.split('[\W_]+')
    data_cl.head()
    return data_cl

def cluster_words(mylist,token_text):
    for row in token_text[['author', 'text']].iterrows():
        r = row[1]
        for word in r.text:
            mylist.append((r.author, word))
    return mylist

def lowersplit(frame,column,stop):
    frame[column] = frame.word.str.lower()
    frame.head()
    
    frame[column] = frame[column].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    frame.head()
    
    frame = frame[frame.word.str.len() > 0]
    frame.head()
    return frame

def pretty_plot_top_n(series, top_n=5, index_level=0):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6),tight_layout=True)
    r = series\
    .groupby(level=index_level)\
    .nlargest(top_n)\
    .reset_index(level=index_level, drop=True)
    r.plot.bar()
    plt.title("top 5 TF-IDF terms")
    plt.ylabel("TF-IDF score")
    plt.savefig('TF-IDF.png')
    return r.to_frame()

def pretty_plot_top_nw(series, top_n=5, index_level=0):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6),tight_layout=True)
    r = series\
    .groupby(level=index_level)\
    .nlargest(top_n)\
    .reset_index(level=index_level, drop=True)
    r.plot.bar()
    plt.title("top 5 most Frequent terms")
    plt.ylabel("Frequency of terms")
    plt.savefig('Count.png')
    return r.to_frame()

def pretty_plot_top_ntf(series, top_n=5, index_level=0):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6),tight_layout=True)
    r = series\
    .groupby(level=index_level)\
    .nlargest(top_n)\
    .reset_index(level=index_level, drop=True)
    r.plot.bar()
    plt.title("top 5 TF terms")
    plt.ylabel("TF score")
    plt.savefig('Term_Frequency.png')
    return r.to_frame()

#def wordcloud(data,column):
#    import matplotlib.pyplot as plt
#    from subprocess import check_output
#    from wordcloud import WordCloud, STOPWORDS
#    stopwords = set(STOPWORDS)
#    wordcloud = WordCloud(
#                          background_color='white',
#                          stopwords=stopwords,
#                          max_words=200,
#                          max_font_size=40, 
#                          random_state=42
#                         ).generate(str(data[column]))
#
#    print(wordcloud)
#    fig = plt.figure(1)
#    plt.imshow(wordcloud)
#    plt.axis('off')
#    plt.show()
#    fig.savefig("word2.png", dpi=900)


