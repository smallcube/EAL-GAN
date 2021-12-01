import torch
import numpy as np
from sklearn.metrics import roc_auc_score

import argparse

def active_sampling(args, real_x, real_y, NetD_Ensemble, need_sample=True):
    if need_sample:
        pt = None
        for i in range(args.ensemble_num):
            netD = NetD_Ensemble[i]
            pt_i = netD(real_x, mode=1)   #get the confidence on real data
            if i==0:
                pt = pt_i.detach()
            else:
                pt += pt_i.detach()
        pt /= args.ensemble_num
        pt = pt.view(pt.shape[0],)
        batch_size_selected = int(real_x.shape[0]*args.active_rate)
        batch_size_selected = max(1, batch_size_selected)
        pt = torch.abs(pt-0.5)  # select the instance with low margin value
        _, idx = torch.sort(pt, descending=False)
        X = real_x[idx[0:batch_size_selected]].detach()
        Y = real_y[idx[0:batch_size_selected]].detach()
        X_unlabeled = real_x[idx[batch_size_selected:]].detach()
        Y = Y.view(Y.shape[0],)
        #X = X.view(X.shape[0], -1)
        return X, Y, X_unlabeled, idx[0:batch_size_selected]
    else:
        batch_size_selected = int(real_x.shape[0]*args.active_rate)
        batch_size_selected = max(1, batch_size_selected)
        X = real_x[0:batch_size_selected].detach()
        Y = real_y[0:batch_size_selected].detach()
        X_unlabeled = None
        idx = torch.tensor(np.arange(batch_size_selected)).view(-1,).long()
        return X, Y, X_unlabeled, idx

def get_gmean(y, y_pred, threshold=0.5):
    """Utility function to calculate precision @ rank n.

    Parameters
    ----------
    y : list or numpy array of shape (n_samples,)
        The ground truth. Binary (0: inliers, 1: outliers).

    y_pred : list or numpy array of shape (n_samples,)
        The raw outlier scores as returned by a fitted model.

    
    Returns
    -------
    Gmean: float
    """
    #y_pred = get_label_n(y, y_pred)
    y = y.reshape(-1, )
    y_pred = y_pred.reshape(-1, )
    y_pred = (y_pred >= threshold).astype('int')
    ones_all = (y==1).sum()
    ones_correct = ((y==1) & (y_pred==1)).sum()
    zeros_all = (y==0).sum()
    zeros_correct = ((y==0) & (y_pred==0)).sum()
    Gmean = np.sqrt((1.0*ones_correct/ones_all) * (1.0*zeros_correct/zeros_all))
    #Gmean *= np.sqrt

    return Gmean

def AUC_and_Gmean(y_test, y_scores):
    #print(y_test)
    #print(y_scores)

    auc = round(roc_auc_score(y_test, y_scores), ndigits=4)
    gmean = round(get_gmean(y_test, y_scores, 0.5), ndigits=4)

    return auc, gmean

def parse_args():
    ################################################################################
    # Settings
    ################################################################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', default='cardio', type=str)
    parser.add_argument('--net_name', default='cardio_mlp', type=str)
    parser.add_argument('--xp_path', default='./log', type=str)
    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--load_config', default=None,
                help='Config JSON-file path (default: None).')
    parser.add_argument('--load_model', default=None,
                help='Model file path (default: None).')
    parser.add_argument('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
    parser.add_argument('--ratio_known_normal', type=float, default=0.9,
                help='Ratio of known (labeled) normal training examples.')
    parser.add_argument('--ratio_known_outlier', type=float, default=0.5,
                help='Ratio of known (labeled) anomalous training examples.')
    parser.add_argument('--ratio_pollution', type=float, default=0.0,
                help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
    parser.add_argument('--seed', type=int, default=0, help='Set seed. If -1, use randomization.')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
    
    parser.add_argument('--n_jobs_dataloader', type=int, default=0,
                help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('--normal_class', type=int, default=0,
                help='Specify the normal class of the dataset (all other classes are considered anomalous).')
    parser.add_argument('--known_outlier_class', type=int, default=1,
                help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
    parser.add_argument('--n_known_outlier_classes', type=int, default=1,
                help='Number of known outlier classes.'
                    'If 0, no anomalies are known.'
                    'If 1, outlier class as specified in --known_outlier_class option.'
                    'If > 1, the specified number of outlier classes will be sampled at random.')
    

    #parameter for GAN
    parser.add_argument('--channel', type=int, default=64,
                        help='capacity of the generator and discriminator')
    parser.add_argument('--resolution', type=int, default=32,
                        help='resolution of the images')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gamma for the scheduler')   
    parser.add_argument('--step_size', type=int, default=10,
                        help='step_size for the scheduler to adjust learning rate')                            
    parser.add_argument('--max_epochs', type=int, default=20,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr_g', type=float, default=0.0001,
                        help='Learning rate of generator.')
    parser.add_argument('--lr_d', type=float, default=0.0001,
                        help='Learning rate of discriminator.')
    parser.add_argument('--active_rate', type=float, default=0.05,
                        help='the proportion of instances that need to be labeled.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size.')
    parser.add_argument('--dim_z', type=int, default=128,
                        help='dim for latent noise.')
    parser.add_argument('--ensemble_num', type=int, default=5,
                        help='the number of dis in ensemble.')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='if GPU used')
    parser.add_argument('--feat_dim', type=int, default=3,
                        help='channel number of images')
    
    return parser.parse_args()
