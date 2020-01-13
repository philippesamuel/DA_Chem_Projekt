#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:02:22 2020

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
from scipy.stats import bartlett
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from factor_analyzer import FactorAnalyzer
import seaborn as sns

df = pd.read_pickle('process_data_after_remove_variabel_remain_96.pkl')

# import dataset
X = df.drop('qc_salzrckhalt', axis = 1)
y = df['qc_salzrckhalt']

# Adequcy Test :  need to evaluate the “factorability” of our dataset. 
# Factorability means "can we found the factors in the dataset?"


# Bartletss`s Test
VarbList = df.columns
chi_square_value,p_value=calculate_bartlett_sphericity(X)
chi_square_value, p_value
# --> p Value = 0 that mean the test was statistically significant, the obvserved correlation matrix is not an identy matrix

# Kaiser_Meyer_Olkin Test
kmo_all,kmo_model=calculate_kmo(X)
kmo_model
# --> KMO value of 0.653 indicates a moderate suitableity for factory analysis  ' Source Cureton, E. E./ D'Agostino, R. B. 1983: Factor analysis: an applied approach. Hillside, NJ: Lawrence Erlbaum Associates, S. 389 f.

# Choosing Number of Factors
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(rotation=None, n_factors=30)
fa.fit(X)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev
# --> only 30 Eigenvalues greater than 1 , so only choose them ?

# Create scree plot 
g = plt.scatter(range(1,X.shape[1]+1),ev)
#g = plt.plot(range(1,X.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

figure = g.get_figure()
figure.savefig('Scree_plot.pdf', dpi=400) 


# Performing Factor Analysis
# Create factor analysis object and perform factor analysis
fa = FactorAnalyzer(rotation='varimax', n_factors= 30)
fa.fit(X)
a = fa.loadings_

# Get variance of each factor
factorVar = fa.get_factor_variance()
factorVar = np.asarray(factorVar)
factorVar.sum(axis = 1)
# --> Total of 60 % Variance is explained by the 30 factors

# Make Faktor plot with named legend, does not work yet
#FA = FactorAnalysis(n_components = 30).fit_transform(X.values)
#a = pd.DataFrame(FA)
#newNames = list(VarbList[0:30])
#oldNames = list(a.columns[0:30])
#
#rename = {i:j for i,j in zip(oldNames,newNames)}
#a.rename(columns = rename, inplace = True)
#
#plt.figure(figsize=(12,8))
#plt.title('Factor Analysis Components')
#FAlist = list(a.columns)


#for i in range(30) :
#    j = i+1
#    if i < 29: 
#        sp = sns.scatterplot(a.iloc[:,i], a.iloc[:,j], hue = y)
#        sp.set(ylim=(-20,100))
#        sp.set(xlim=(-20,100))
#        
#    else:
#       sp = sns.scatterplot(a.iloc[:,i-1], a.iloc[:,0], hue = y)
#       sp.set(ylim=(-20,100))
#       sp.set(xlim=(-20,100))
#   




from sklearn.decomposition import TruncatedSVD 
svd = TruncatedSVD(n_components=30, random_state=42).fit_transform(X.values)

plt.figure(figsize=(12,8))
plt.title('SVD Components')
plt.scatter(svd[:,0], svd[:,1])
plt.scatter(svd[:,1], svd[:,2])
plt.scatter(svd[:,2],svd[:,3])
plt.scatter(svd[:,3],svd[:,4])
plt.scatter(svd[:,4],svd[:,5])
plt.scatter(svd[:,5],svd[:,6])
plt.scatter(svd[:,6],svd[:,7])
plt.scatter(svd[:,7],svd[:,8])
plt.scatter(svd[:,8],svd[:,9])
plt.scatter(svd[:,9],svd[:,0])