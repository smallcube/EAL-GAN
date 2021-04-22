# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import os
import sys
from time import time

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

import torch

from models.pyod_utils import standardizer, AUC_and_Gmean
from models.pyod_utils import precision_n_scores, gmean_scores
from models.utils import parse_args
from sklearn.metrics import roc_auc_score
from models.CB_GAN import CB_GAN

# Define data file and read X and y
folder_name = 'data_mat'
save_dir = 'results'


mat_file_list = ['arrhythmia.mat',
                 'cardio.mat',
                 'glass.mat',
                 'ionosphere.mat',
                 'letter.mat',
                 'lympho.mat',
                 'mnist.mat',
                 'musk.mat',
                 'optdigits.mat',
                 'pendigits.mat',
                 'pima.mat',
                 'satellite.mat',
                 'satimage-2.mat',
                 'shuttle.mat',
                 'vowels.mat'
                 ]

'''
mat_file_list = [#'single-cluster.mat',
                'multi-cluster.mat',
                'multi-shape.mat']
'''

# define the number of iterations
n_ite = 10
n_classifiers = 1

df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
            'CB-GAN']

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
gmean_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)
args = parse_args()

for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    mat = loadmat(os.path.join(folder_name, mat_file))

    X = mat['X']
    y = mat['y'].ravel()
    y = y.astype(np.long)

    outliers_fraction = np.count_nonzero(y) / len(y)
    print(outliers_fraction)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    gmean_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file[:-4], X.shape[0], X.shape[1], outliers_percentage]

    roc_mat = np.zeros([n_ite, n_classifiers])
    prn_mat = np.zeros([n_ite, n_classifiers])
    gmean_mat = np.zeros([n_ite, n_classifiers])
    time_mat = np.zeros([n_ite, n_classifiers])

    for i in range(n_ite):
        print("\n... Processing", mat_file, '...', 'Iteration', i + 1)
        random_state = np.random.RandomState(i)

        # 60% data for training and 40% for testing
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.4, random_state=random_state)

        # standardizing data for processing

        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        X_train_pandas = pd.DataFrame(X_train_norm)
        X_test_pandas = pd.DataFrame(X_test_norm)
        X_train_pandas.fillna(X_train_pandas.mean(), inplace=True)
        X_test_pandas.fillna(X_train_pandas.mean(), inplace=True)
        X_train_norm = X_train_pandas.values
        X_test_norm = X_test_pandas.values
        
        #X_train_norm, X_test_norm = X_train, X_test
        data_x = torch.from_numpy(X_train_norm).float()
        data_y = torch.from_numpy(y_train).long()
        test_x = torch.from_numpy(X_test_norm).float()
        #test_y = torch.from_numpy(y_test).long()
        #print(data_x)

        t0 = time()
        #todo: add my method
        cb_gan = CB_GAN(args, data_x, data_y, test_x, y_test)
        cb_gan.fit()
        test_scores = cb_gan.predict(test_x)
        train_scores = cb_gan.predict(data_x)
        #clf.fit(X_train_norm)
        #test_scores = clf.decision_function(X_test_norm)
        t1 = time()
        duration = round(t1 - t0, ndigits=4)
        
        roc, prn, gmean = AUC_and_Gmean(y_test, test_scores)
        roc_t, prn_t, gmean_t = AUC_and_Gmean(y_train, train_scores)
        
        print('AUC:{roc}, precision @ rank n:{prn}, Gmean:{gmean}  train_AUC:{train_auc}  train_gmean:{train_gmean}  '  
                    'execution time: {duration}s'.format(roc=roc, prn=prn, gmean=gmean, train_auc=roc_t, train_gmean=gmean_t, duration=duration))

        time_mat[i, 0] = duration
        roc_mat[i, 0] = roc
        prn_mat[i, 0] = prn
        gmean_mat[i, 0] = gmean

    time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)

    roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)

    gmean_list = gmean_list + np.mean(gmean_mat, axis=0).tolist()
    temp_df = pd.DataFrame(gmean_list).transpose()
    temp_df.columns = df_columns
    gmean_df = pd.concat([gmean_df, temp_df], axis=0)

    # Save the results for each run
    save_path1 = os.path.join(save_dir, 'AUC_CB_GAN_'+ folder_name +".csv")
    save_path2 = os.path.join(save_dir, 'Gmean_CB_GAN_'+ folder_name +".csv")
    roc_df.to_csv(save_path1, index=False, float_format='%.3f')
    gmean_df.to_csv(save_path2, index=False, float_format='%.3f')