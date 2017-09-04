#The paper for this file is :
#C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by
#Orthogonal Matching Pursuit, CVPR 2016.

import numpy as np
import time
import math as mt
import random
from scipy.io import loadmat
from genSubspace import *
from sklearn import cluster

ambient_dim = 9
num_subspace = 5
dim_subspace = 6
#num_points = 50
sigma = 0.00

num_experiment = 20

#print(np.__version__)

#Anonymous functions go here
#buildRepresentation = lambda
#genLabel =
####################

num_points_list = np.logspace(np.log10(30), np.log10(20000), 12)
#print(num_points_list)
#print(type(num_points_list))
#print(num_points_list.shape)
num_points_list = num_points_list[0:1]
for ii in range(num_points_list.shape[0]):
    num_points = int(mt.ceil(num_points_list[ii]))
    #print(type(num_points))
    results = np.zeros((num_experiment,6))
    for iExperiment in range(num_experiment):
        random.seed(iExperiment)
        #if len(num_points) == 1:
        num_points = num_points*np.ones([1,num_subspace], dtype = np.int64)
        X, s = genSubspace(ambient_dim, num_subspace, num_points, dim_subspace, sigma);
        N = np.sum(num_points)
        mat = loadmat("test_data.mat")
        ###########################################################
        X = mat['X']
        s = mat['s']
        s = s[0]
        ###########################################################
        time0 = time.time()
        R = Ompmat(X, 6, 1e-6)
        time1 = time.time()
        R = R.toarray()
        np.fill_diagonal(R,0)
        A = np.abs(R) + (np.abs(R)).transpose()   #check this
        spectral_var = cluster.SpectralClustering(n_clusters = num_subspace)
        spectral_var.fit(A)
        #groups = spectral_var.labels_.astype(np.int)
        mat1 = loadmat("pred_labels.mat")
        groups = mat1['groups']
        time2 = time.time()
        perc, vec = evalSSR_perc(R,s)
        ssr = evalSSR_error(R,s)
        conn = evalConn(A,s)
        accr = evalAccuracy(s, groups)

        dataValue = [iExperiment, perc, ssr, conn, accr, time1, time2]

        results[iExperiment,:] = dataValue[2:]

        ################Check routine#####################
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = np.array([0,0,1])
        print(evalSSR_perc(X,labels))
        ##################################################
    dataformat = "Ni = {} perc = {}, ssr = {}, conn = {}, accr = {}, time1 = {}, time2 = {}"
    dataValue = [Ni[1], mean(results, 1)]
    print(dataformat.format(dataValue))