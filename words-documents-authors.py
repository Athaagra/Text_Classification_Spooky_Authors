# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:40:52 2018

@author: Kel3vra
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['font.size'] = 15
N = 3
EAP = (7900, 201493, 96147)
#menStd =   (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.2       # the width of the bars

fig = plt.figure(tight_layout=True)
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, EAP, width, color='royalblue', )

HPL = (5635, 157376, 80959)
#womenStd =   (3, 5, 2, 3, 3)
rects2 = ax.bar(ind+width+0.02, HPL, width, color='seagreen')

MWS = (6044, 166070, 78864)
#womenStd =   (3, 5, 2, 3, 3)
rects3 = ax.bar(ind+width+width+0.05, MWS, width, color='red')

# add some
ax.set_ylabel('Overall')
ax.set_title('Number of Words and Docs')
ax.set_xticks(ind + width / 3)
ax.set_xticklabels( ('Doc', 'Words', 'No stopwords') )
ax.legend( (rects1[0], rects2[0],rects3[0]), ('EAP', 'HPL','MWS') )
plt.xlabel('\nTotal: 19576, Words: 524939, Words_sw: 255970 ')
plt.savefig('authors_Last.png')
plt.show()