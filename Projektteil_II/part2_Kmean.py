#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:11:05 2020

@author: elron
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
import warnings
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection)
warnings.filterwarnings("ignore")
from sklearn.decomposition import FactorAnalysis

df = pd.read_pickle('process_data_after_remove_variabel_remain_96.pkl')



# import dataset
X = df.drop('qc_salzrckhalt', axis = 1)
y = df['qc_salzrckhalt']


# choose optimal number of clusters via elbow method
from sklearn.cluster import KMeans
wscc = []  #Wihin clusters of sum of squares
for i in range(1,11):
    kmean = KMeans(n_clusters=i, 
                   init = 'k-means++', 
                   max_iter=300, 
                   n_init=30, 
                   random_state = 0)
    kmean.fit(X)
    wscc.append(kmean.inertia_)
plt.figure(figsize=(20,12))
g = plt.plot(range(1,11),wscc)
plt.title('The Elbow Method')
plt.xlabel('Number of cluster')
plt.ylabel('WCSS')
plt.show()

# Applying kmeans with right number of clusters
kmeans = KMeans(n_clusters=2, 
                init = 'k-means++',
                max_iter= 300,
                n_init=30,
                random_state= 0)

y_means = kmeans.fit_predict(X)









