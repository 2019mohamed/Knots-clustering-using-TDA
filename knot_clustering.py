# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:11:48 2021

@author: M
"""

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


#cmap = get_cmap(1000000)
knots = []
for index,i in enumerate([4000,5000,6000,7000]):
    for j in range(5):
        X = np.random.randn(i,3) / 1000
        X[:,0] += np.cos(np.arange(i)*2*np.pi/i)
        X[:,1] += np.sin(np.arange(i)*2*np.pi/i)
        Z = TSNE(n_jobs=-1, init='random', random_state=np.random.randint(5,42)).fit_transform(X)
        plt.scatter(Z[:,0] , Z[:,1] , c = np.random.rand(3,))
        plt.show()
        knots.append(Z)
        
homology_dimensions = (0, 1)
VR = VietorisRipsPersistence(
    metric='euclidean', homology_dimensions=homology_dimensions)

# Array of persistence diagrams, one per point cloud in the input
diagrams = VR.fit_transform(knots)

PE = PersistenceEntropy()
F = PE.fit_transform(diagrams)

C = AgglomerativeClustering(n_clusters=5).fit_transform(X)
print(C.labels_)


