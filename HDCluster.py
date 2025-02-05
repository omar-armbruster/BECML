#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Class for clustering without using dimensional reduction techniques.

import numpy as np
import math
import random
from scipy import constants
import sys
from sklearn.cluster import SpectralClustering

class HDCluster:
    def __init__(
        self,
        n_clusters: int,
        gamma: float, 
        ) -> None:
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.affinity_matrix = np.array([])
        self.isFit = False
    
    def HDAffinity(self, data, gamma):
        """
        Computes a radial basis function affinity matrix without dimensionally reducing the data. Input gamma as the scaling coeffcient.
        Inputs: 
        data: An array of nxn numpy arrays
        gamma: The scaling coefficient for a gaussian distance function 
        """
        dist_matrix = np.array([[ np.sum((x - y)**2) for y in data ]for x in data]) #Computes matrix with distance between images squared
        aff_matrix = np.exp(-gamma*dist_matrix) #Affinity matrix using gaussian distance
        self.affinity_matrix = aff_matrix
        return aff_matrix
        
    def fit_predict(self, data):
        """
        Computes clusters for a given dataset (data) using spectral clustering based on the high definition affinity matrix.
        Inputs:
        data: An array of nxn numpy arrays
        """
        aff_matrix = self.HDAffinity(data, self.gamma)
        clusters = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed').fit_predict(aff_matrix)
        self.isFit = True
        self.clusters = clusters
        return clusters
    
    def cluster_accuracy(self, labels, binning = False, bin_num = 75, bin_range = (0.5, 2), xlabels = np.array([])):
        """
        Calculates the accuracy of a given set of cluster labels.
        Inputs:
        labels: List containing the correct labels for the clustered data 
        binning: A boolean determining if the results are binned. The following parameters are only used if binning = True
        bin_num: An int representing the number of bins for the data to be sorted into
        bin_range: The minimum and maximum values of the bins 
        xlabels: The values to be binned 
        """
        if not self.isFit:
            raise Exception('Object must be trained before calculating accuracy!')
        accuracy = []
        for i, label in enumerate(labels):
            if (label == self.clusters[i]):
                accuracy.append(1)
            else:
                accuracy.append(0)
        if self.n_clusters == 2:
            accuracy = ((np.array(accuracy))-1)*-1 if sum(accuracy) < 0.5*len(accuracy) else np.array(accuracy) #Shortcut to maximize accuracy
        else: 
            accuracy = np.array(accuracy)

        self.accuracy = accuracy
        if not binning:
            return accuracy
        else:
            if len(xlabels) != len(accuracy):
                raise Exception('x labels must be the same length as the input data for binning')
            return self.binAccuracy(bin_num, xlabels, accuracy, bin_range)
        
    def calculate_conductances(self, binning = False, bin_num = 75, bin_range = (0.5, 2), xlabels = np.array([])):
        """
        Calculates the intracluster conductance of the dataset
        Inputs:
        labels: List containing the correct labels for the clustered data 
        binning: A boolean determining if the results are binned. The following parameters are only used if binning = True
        bin_num: An int representing the number of bins for the data to be sorted into
        bin_range: The minimum and maximum values of the bins 
        xlabels: The values to be binned
        """
        if not self.isFit:
            raise Exception('Object must be trained before calculating conductance!')
        if self.n_clusters != 2:
            raise Exception('Function only calculates conductance for two clusters')
        conductance0 = []
        conductance1 = []
        for v in range(len(self.clusters)):
            vol = np.sum(self.affinity_matrix[v])
            cutWeights0 = 0
            cutWeights1 = 0
            for w in range(len(self.clusters)):
                if self.clusters[w] == 0:
                    cutWeights0 += self.affinity_matrix[v,w]
                elif self.clusters[w] == 1:
                    cutWeights1 += self.affinity_matrix[v,w]
            conductanceScore0 = cutWeights0/vol if vol > 0 else 1
            conductanceScore1 = cutWeights1/vol if vol > 0 else 1
            conductance0.append(conductanceScore0)
            conductance1.append(conductanceScore1)
        if not binning:
            return conductance0, conductance1
        else:
            if len(xlabels) != len(conductance0):
                raise Exception('x labels must be the same length as the input data for binning')
            binLab0, binConductance0 = self.binAccuracy(bin_num, xlabels, conductance0, bin_range)
            _, binConductance1 = self.binAccuracy(bin_num, xlabels, conductance1, bin_range)
            return binLab0, binConductance0, binConductance1

    def binAccuracy(self, bin_num, xlabels, ylabels, bin_range):
        """
        bin_num: The number of bins to be sorted 
        xlabels: The labels corresponding to the y value to be binned
        ylabels: The values to be binned
        Returns an array of bins and an array of tuples containing the mean and standard deviation of the accuracy per bin.
        """
        bins = np.linspace(bin_range[0], bin_range[1], bin_num) ## Hardcoded for 0.5Tc and 2Tc. Change if dataset has different bounds
        xbins = np.digitize(xlabels, bins)
        binVals0 = {}
        for i in range(len(xbins)):
            if bins[xbins[i]] in list(binVals0.keys()):
                binVals0[bins[xbins[i]]].append(ylabels[i])
            else:
                binVals0[bins[xbins[i]]] = [ylabels[i]]
        for key in binVals0.keys():
            m = np.mean(binVals0[key])
            std = np.std(binVals0[key])
            binVals0[key] = [m, std]
        return np.array(list(binVals0.keys())), np.array(list(binVals0.values()))

  