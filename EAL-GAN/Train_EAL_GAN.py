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
from models.EAL_GAN import EAL_GAN

if __name__=='__main__':
    # Define data file and read X and y
    data_root = './data'
    save_dir = './results'


    data_names = [#'arrhythmia.mat',
                  #  'cardio.mat',
                  #  'glass.mat',
                  #  'ionosphere.mat',
                  #  'letter.mat',
                  #  'lympho.mat',
                  #  'mnist.mat',
                  #  'musk.mat',
                  #  'optdigits.mat',
                  #  'pendigits.mat',
                  #  'pima.mat',
                  #  'satellite.mat',
                  #  'satimage-2.mat',
                  #  'shuttle.mat',
                  #  'vowels.mat',
                  #  'annthyroid.mat',
                  #  'campaign.mat',
                  #  'celeba.mat',
                    'fraud.mat',
                    'donors.mat'
                    ]
    # define the number of iterations
    n_ite = 10
    n_classifiers = 1

    df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc',
                'AUC_Mean', 'AUC_Std', 'Gmean', 'Gmean_Std']

    # initialize the container for saving the results
    roc_df = pd.DataFrame(columns=df_columns)
    gmean_df = pd.DataFrame(columns=df_columns)
    anomaly_ratio_df = pd.DataFrame(columns=df_columns)
    overall_ratio_df = pd.DataFrame(columns=df_columns)
    time_df = pd.DataFrame(columns=df_columns)
    args = parse_args()

    for data_name in data_names:
        mat = loadmat(os.path.join(data_root, data_name))

        X = mat['X']
        y = mat['y'].ravel()
        y = y.astype(np.long)
        idx_norm = y == 0
        idx_out = y == 1

        outliers_fraction = np.count_nonzero(y) / len(y)
        print(outliers_fraction)
        outliers_percentage = round(outliers_fraction * 100, ndigits=4)

        # construct containers for saving results
        roc_list = [data_name[:-4], X.shape[0], X.shape[1], outliers_percentage]
        gmean_list = [data_name[:-4], X.shape[0], X.shape[1], outliers_percentage]
        anomaly_ratio_list = [data_name[:-4], X.shape[0], X.shape[1], outliers_percentage]
        overall_ratio_list = [data_name[:-4], X.shape[0], X.shape[1], outliers_percentage]
        
        roc_mat = np.zeros([n_ite, n_classifiers])
        gmean_mat = np.zeros([n_ite, n_classifiers])
        anomaly_ratio_mat = np.zeros([n_ite, n_classifiers])
        overall_ratio_mat = np.zeros([n_ite, n_classifiers])

        for i in range(n_ite):
            print("\n... Processing", data_name[:-4], '...', 'Iteration', i + 1)
            random_state = np.random.RandomState(i)

            # 60% data for training and 40% for testing; keep outlier ratio
            
            X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],
                                                                                    test_size=0.4,
                                                                                    random_state=random_state)
            X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
                                                                                test_size=0.4,
                                                                                random_state=random_state)
            X_train = np.concatenate((X_train_norm, X_train_out))
            X_test = np.concatenate((X_test_norm, X_test_out))
            y_train = np.concatenate((y_train_norm, y_train_out))
            y_test = np.concatenate((y_test_norm, y_test_out))
            
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
            test_y = torch.from_numpy(y_test).long()
            #print(data_x)

            t0 = time()
            #todo: add my method
            eal_gan = EAL_GAN(args, data_x, data_y, test_x, test_y)
            best_auc, best_gmean = eal_gan.fit()
            
            t1 = time()
            duration = round(t1 - t0, ndigits=4)
            
            
            print('AUC:%.4f, Gmean:%.4f  execution time: %.4f s' % (best_auc, best_gmean, duration))

            roc_mat[i, 0] = best_auc
            gmean_mat[i, 0] = best_gmean

        roc_list = roc_list + np.mean(roc_mat, axis=0).tolist() + np.std(roc_mat, axis=0).tolist() + np.mean(gmean_mat, axis=0).tolist() + np.std(gmean_mat, axis=0).tolist()
        temp_df = pd.DataFrame(roc_list).transpose()
        temp_df.columns = df_columns
        roc_df = pd.concat([roc_df, temp_df], axis=0)

        
        # Save the results for each run
        save_path1 = os.path.join(save_dir, "AUC_EAL_GAN.csv")
        
        roc_df.to_csv(save_path1, index=False, float_format='%.4f')