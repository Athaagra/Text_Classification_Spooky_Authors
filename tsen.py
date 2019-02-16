# -*- coding: utf-8 -*-
"""
Created on Thu May 17 00:14:05 2018

@author: Kel3vra
"""

from sklearn.decomposition import PCA
from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create the visualizer and draw the vectors
tsne = TSNEVisualizer()
tsne.fit(tf_matrix, y)
tsne.poof()