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

from src.my_utils import parse_args
from src.EAL_GAN import EAL_GAN
from src.datasets.main import load_dataset

if __name__=='__main__':
    # Define data file and read X and y
    data_root = './data'
    save_dir = './results'
    n_ite = 10

    data_names = [#'mnist',
                'fmnist',
                #'cifar10',
                ]
    data_resolution = {'mnist':32, 'fmnist':32, 'cifar10':32}
    # define the number of iterations
    df_columns = ['Data', 'auc_mean', 'auc_std', 'gmean', 'gmean_std']

    # initialize the container for saving the results
    roc_df = pd.DataFrame(columns=df_columns)
    gmean_df = pd.DataFrame(columns=df_columns)
    
    args = parse_args()
    roc_df = pd.DataFrame(columns=df_columns)
    gmean_df = pd.DataFrame(columns=df_columns)
    outlier_class = 0

    

    for data_name in data_names:
        roc_list = [data_name]
        gmean_list = [data_name]
        
        roc_mat = np.zeros([1, 1])
        gmean_mat = np.zeros([1, 1])

        args.feat_dim = 3 if data_name=='cifar10' else 1
        #for i in range(0, n_ite):
        dataset = load_dataset(data_name, data_root, args.normal_class, outlier_class, args.n_known_outlier_classes,
                        args.ratio_known_normal, args.ratio_known_outlier, args.ratio_pollution,
                        random_state=args.seed, resolution=data_resolution[data_name])

        
        #todo: add my method
        cb_gan = EAL_GAN(args, dataset)
        best_auc, best_gmean = cb_gan.fit()
    
        print('AUC:%.4f, Gmean:%.4f ' % (best_auc, best_gmean))

        roc_mat[0, 0] = best_auc
        gmean_mat[0, 0] = best_gmean

        roc_list = roc_list + np.mean(roc_mat, axis=0).tolist() + np.std(roc_mat, axis=0).tolist() + np.mean(gmean_mat, axis=0).tolist()+np.std(gmean_mat, axis=0).tolist()
        temp_df = pd.DataFrame(roc_list).transpose()
        temp_df.columns = df_columns
        roc_df = pd.concat([roc_df, temp_df], axis=0)

        # Save the results for each run
        save_path1 = os.path.join(save_dir, "AUC_EAL_GAN.csv")
        roc_df.to_csv(save_path1, index=False, float_format='%.4f')
        