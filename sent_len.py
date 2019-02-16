# -*- coding: utf-8 -*-
"""
Created on Wed May 23 16:29:31 2018

@author: Kel3vra
"""


import nltk
import numpy
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import inaugural
stopwords = set(STOPWORDS) 
sent_len = []
text =data_cl['text'].values.tolist()
author = data_cl['author'].values.tolist()

clear_list = []
for i in range(len(text)):
    clear_list.append([text[i],author[i]])

total_lens = 0
i=0
sent =0
for i, sent in enumerate(text):
   total_lens += len(sent)
total_lens
avg_sent_len = total_lens / i
avg_sent_len
HPL=[]
for i in range (len(clear_list)):
    if clear_list[i][1]=='HPL':
        HPL.append(clear_list[i][0])
sent_len = []
for i in range(len(MWS)):
    sen = MWS[i][0][0]
    sent_w1 = nltk.word_tokenize(sen)
    sent_w2 = [sen for sen in sent_w1 if i not in STOPWORDS]
    sent_len.append(len(sent_w2))
MWS_len=[]
for i in range (len(MWS)):
    MWS_len.append(len(MWS[i][0]))

xo =  np.array(EAP_len)
np.where(xo == 140)








plt.hist(MWS_len, bins=120, rwidth = 0.5) 
#plt.axis([50, 110, 0, 0.06]) 
#axis([xmin,xmax,ymin,ymax])
plt.xlim(0,170)
plt.xlabel('Words')
plt.ylabel('Sentences')
plt.savefig('MWS_Histogram.png')
plt.show()



import pandas as pd

dfh = pd.DataFrame(EAP_len)
dfh.hist(bins = 50,rwidth = 0.5);