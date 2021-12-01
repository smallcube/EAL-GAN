import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from numpy import *
import argparse

def load_data_V2(data_name):
    #data_path = os.path.join('./data/', data_name)
    data_path = data_name
    data = pd.read_table('{path}'.format(path = data_path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1).astype('category')
    data_x = data.values
    
    #data_y = y.cat.codes.values
    #print(data_x.size())
    data_y = np.zeros((data_x.shape[0],) ,dtype = np.int)
    idx = (y.values=='out')
    data_y[idx] = 1
    min_label = 1
    #min_label = 0 if zeros_counts<ones_counts else 1
    #class_mapping = {label:idx for idx,label in enumerate(set(y.values))}
    #data_y = y.map(class_mapping).values

    n_classes = int(max(data_y)+1)
    return data_x, data_y, min_label, n_classes

def load_data(data_name):
    data_path = os.path.join('./data/', data_name)
    data = pd.read_table('{path}'.format(path = data_path), sep=',', header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    id = data.pop(0)
    y = data.pop(1).astype('category')
    data_x = data.values
    data_y = y.cat.codes.values
    zeros_counts = (data_y==0).sum()
    ones_counts = (data_y==1).sum()
    min_label = 0 if zeros_counts<ones_counts else 1
    
    n_classes = int(max(data_y)+1)
    #print("minLabel_f=%d" % (min_label))
    return data_x, data_y, min_label, n_classes

class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):    
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)    
        # return self.variable
    
    # Silly hack: overwrite the to() method to wrap the new object
    # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)    
        return new_obj

def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', 
                fp16=False,z_var=1.0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=z_var)
    z_ = z_.to(device,torch.float16 if fp16 else torch.float32)   
  
    if fp16:
        z_ = z_.half()

    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical',num_categories=nclasses)
    y_ = y_.to(device, torch.int64)
    return z_, y_

'''
def sample_selector(args, z, y, real_x, real_y, min_label, netG, NetD_Ensemble, low_margin=0.4, high_margin=0.8):
    idx = (real_y==min_label)
    extra_batch_size = real_y.shape[0]-int(idx.sum())*2
    count = 0
    X = real_x.detach()
    Y = real_y.detach()
    while count<extra_batch_size:
        z.sample_()
        y.sample_()
        generated_y = torch.ones_like(y)*min_label
        generated_x = netG(z, generated_y)
        generated_x = generated_x.detach()
        #print(generated_x.shape)
        #compute the margin
        pt = None
        for i in range(args.ensemble_num):
            netD = NetD_Ensemble[i]
            pt_i = netD(generated_x, generated_y, fine_tuning=True)
            if i==0:
                pt = pt_i.detach()
            else:
                pt += pt_i.detach()
        pt /= args.ensemble_num
        #print(pt.shape)
        pt = pt.view(pt.shape[0],)
        #print(pt[0])
        #index =  (pt>=low_margin) & (pt<=high_margin)
        index =  (pt>=low_margin)
        #print(index)
        
        index = index.view(index.shape[0],)
        #print(index.shape)
        selected_size = int(index.sum())
        tempX = generated_x[index]
        #print(tempX)
        tempY = generated_y[index]
        if count+ selected_size< extra_batch_size:
            count += selected_size
            X = torch.cat([X, tempX], 0)
            Y = torch.cat([Y, tempY], 0)
        else:
            count += selected_size
            X = torch.cat([X, tempX[0:(extra_batch_size-count)]], 0)
            Y = torch.cat([Y, tempY[0:(extra_batch_size-count)]], 0)
    return X, Y
'''

def sample_selector(args, z, y, real_x, real_y, min_label, netG, NetD_Ensemble, low_margin=0.4, high_margin=0.8):
    idx = (real_y==min_label)
    extra_batch_size = real_y.shape[0]-int(idx.sum())*2
    X = real_x.detach()
    Y = real_y.detach()
    z.sample_()
    y.sample_()
    generated_y = torch.ones_like(y)*min_label
    generated_x = netG(z, generated_y)
    generated_x = generated_x.detach()
    #print(generated_x.shape)
    #compute the margin
    pt = None
    for i in range(args.ensemble_num):
        netD = NetD_Ensemble[i]
        pt_i = netD(generated_x, generated_y, mode=2)
        if i==0:
            pt = pt_i.detach()
        else:
            pt += pt_i.detach()
    pt /= args.ensemble_num
    pt = pt.view(pt.shape[0],)

    _, idx = torch.sort(pt, descending=True)
    tempX = generated_x[idx[0:extra_batch_size]].detach()
    tempY = generated_y[idx[0:extra_batch_size]].detach()
    
    X = torch.cat([X, tempX], 0)
    Y = torch.cat([Y, tempY], 0)
    return X, Y

def sample_selector_V1(args, z, y, real_x, real_y, min_label, netG, NetD_Ensemble, low_margin=0.4, high_margin=0.8):
    idx = (real_y==min_label)
    extra_batch_size = real_y.shape[0]-int(idx.sum())*2
    X = real_x.detach()
    Y = real_y.detach()
    z.sample_()
    y.sample_()
    generated_y = torch.ones_like(y)*min_label
    generated_x = netG(z, generated_y)
    generated_x = generated_x.detach()
    #print(generated_x.shape)
    #compute the margin
    pt = None
    for i in range(args.ensemble_num):
        netD = NetD_Ensemble[i]
        pt_i = netD(generated_x, generated_y, fine_tuning=True)
        if i==0:
            pt = pt_i.detach()
        else:
            pt += pt_i.detach()
    pt /= args.ensemble_num
    pt = pt.view(pt.shape[0],)

    _, idx = torch.sort(pt, descending=True)
    tempX = generated_x[idx[0:extra_batch_size]].detach()
    tempY = generated_y[idx[0:extra_batch_size]].detach()
    
    X = torch.cat([X, tempX], 0)
    Y = torch.cat([Y, tempY], 0)
    return X, Y


def active_sampling_V1(args, real_x, real_y, NetD_Ensemble, need_sample=True):
    if need_sample:
        pt = None
        for i in range(args.ensemble_num):
            netD = NetD_Ensemble[i]
            pt_i = netD(real_x, mode=2)   #get the confidence on real data
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
        #print(real_x.shape)
        #print("Xhspae=",X.shape)
        #print("unlable=", X_unlabeled.shape)
        #Y2 = real_y[idx[-batch_size_selected:]]
        #sum1 = (Y==1).sum()
        #sum2 = (Y2==1).sum()
        #sum3 = (real_y==1).sum()
        #print("sum1=%d   sum2=%d   sum3=%d"%(sum1, sum2, sum3))
        Y = Y.view(Y.shape[0],)
        X = X.view(X.shape[0], -1)
    else:
        batch_size_selected = int(real_x.shape[0]*args.active_rate)
        X = real_x[0:batch_size_selected].detach()
        Y = real_y[0:batch_size_selected].detach()
        X_unlabeled = None
        #sum1 = (Y==1).sum()
        #print(sum1)
    return X, Y, X_unlabeled

def active_sampling(args, real_x, real_y, NetD_Ensemble, need_sample=True):
    if need_sample:
        pt = None
        for i in range(args.ensemble_num):
            netD = NetD_Ensemble[i]
            pt_i = netD(real_x, mode=2)   #get the confidence on real data
            if i==0:
                pt = pt_i.detach()
            else:
                pt += pt_i.detach()
        pt /= args.ensemble_num
        pt = pt.view(pt.shape[0],)
        batch_size_selected = int(real_x.shape[0]*args.active_rate)
        pt = torch.abs(pt-0.5)  # select the instance with low margin value
        _, idx = torch.sort(pt)
        X = real_x[idx[0:batch_size_selected]].detach()
        Y = real_y[idx[0:batch_size_selected]].detach()
        X_unlabeled = real_x[idx[batch_size_selected:]].detach()
        #Y2 = real_y[idx[-batch_size_selected:]]
        #sum1 = (Y==1).sum()
        #sum2 = (Y2==1).sum()
        #sum3 = (real_y==1).sum()
        #print("sum1=%d   sum2=%d   sum3=%d"%(sum1, sum2, sum3))
        Y = Y.view(Y.shape[0],)
        X = X.view(X.shape[0], -1)
    else:
        batch_size_selected = int(real_x.shape[0]*args.active_rate)
        X = real_x[0:batch_size_selected].detach()
        Y = real_y[0:batch_size_selected].detach()
        X_unlabeled = None
        #sum1 = (Y==1).sum()
        #print(sum1)
    return X, Y, X_unlabeled


def loss_gen(input, weights=None, gamma=2.0):
    logpt = F.softplus(-input)
    pt = torch.exp(logpt)
    
    if weights is None:
        p = pt*1
        p = p.view(len(input), 1)
        p = (1-p)**gamma
        loss = p * logpt
        weights = pt.detach()
        #loss = logpt
    else:
        weights = torch.cat([weights, pt], 1)
        #p,_ = torch.min(instance_weight, 1, keepdim=True)
        p = torch.mean(weights, 1)
        p = p.view(len(input), 1)
        p = (1-p)**gamma
        loss = p*logpt
    
    loss = torch.mean(loss)
    return loss, weights

def loss_dis_real(dis_real, weights=None, gamma=2.0):
    logpt = F.softplus(-dis_real)
    pt = torch.exp(-logpt)
    if weights is None:
        p = pt*1
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p * logpt
        weights = pt.detach()
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p*logpt
    
    loss = torch.mean(loss)
    return loss, weights

def loss_dis_fake(dis_fake, weights=None, gamma=2.0):
    logpt = F.softplus(dis_fake)
    pt = torch.exp(-logpt)

    if weights is None:
        p = pt*1
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p * logpt
        weights = pt.detach()
        #loss = logpt
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p*logpt
    
    loss = torch.mean(loss)
    
    return loss, weights

def loss_for_real(output_real_fake, output_category, y, real_fake_weight=None, 
                    category_weight=None, gamma=1.0, weights=None):
    '''
    output_real_fake: the output used for predicting whether the input is real or fake(dims:[batch, 1], range in [0,1]
    output_category: used for predicting the category, (dims: [batch, n_classes]
    y: the class label
    '''
    #step 1: loss for real_fake
    batch_size = output_real_fake.size(0)
    pt_real_fake = output_real_fake
    logpt_real_fake = -torch.log(pt_real_fake)
    if real_fake_weight is None:
        real_fake_weight = pt_real_fake.detach()
        p = pt_real_fake*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p * logpt_real_fake
    else:
        real_fake_weight = torch.cat([real_fake_weight, pt_real_fake], 1)
        p = torch.mean(real_fake_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p*logpt_real_fake
    loss_real_fake = torch.mean(loss_real_fake)

    #Step 2: loss for category
    logpt_category = F.log_softmax(output_category, dim=1)
    pt_category = torch.exp(logpt_category)
    index = y.view(batch_size, 1).long()
    pt_category = pt_category.gather(1, index)

    if category_weight is None:
        category_weight = pt_real_fake.detach()
        p = pt_category*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        category_weight = torch.cat([category_weight, pt_category], 1)
        p = torch.mean(category_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_category = p*logpt_category
    loss_category = F.nll_loss(logpt_category, y, weights)

    return loss_real_fake, loss_category, real_fake_weight, category_weight

def loss_for_fake(output_real_fake, output_category, y, real_fake_weight=None, 
                    category_weight=None, gamma=1.0, weights=None):
    '''
    output_real_fake: the output used for predicting whether the input is real or fake(dims:[batch, 1], range in [0,1]
    output_category: used for predicting the category, (dims: [batch, n_classes]
    y: the class label
    '''
    #step 1: loss for real_fake
    batch_size = output_real_fake.size(0)
    pt_real_fake = 1.0 - output_real_fake
    logpt_real_fake = -torch.log(pt_real_fake)
    if real_fake_weight is None:
        real_fake_weight = pt_real_fake.detach()
        p = pt_real_fake*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p * logpt_real_fake
    else:
        real_fake_weight = torch.cat([real_fake_weight, pt_real_fake], 1)
        p = torch.mean(real_fake_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p*logpt_real_fake
    loss_real_fake = torch.mean(loss_real_fake)

    #Step 2: loss for category
    logpt_category = F.log_softmax(output_category, dim=1)
    pt_category = torch.exp(logpt_category)
    index = y.view(batch_size, 1).long()
    pt_category = pt_category.gather(1, index)
    #print(pt_category)

    if category_weight is None:
        category_weight = pt_real_fake.detach()
        p = pt_category*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        category_weight = torch.cat([category_weight, pt_category], 1)
        p = torch.mean(category_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_category = p*logpt_category
    loss_category = F.nll_loss(logpt_category, y, weights)

    return loss_real_fake, loss_category, real_fake_weight, category_weight


def loss_for_realV2(output_real_fake, output_category, y, real_fake_weight=None, 
                    category_weight=None, gamma=1.0, weights=None):
    '''
    output_real_fake: the output used for predicting whether the input is real or fake(dims:[batch, 1], range in [0,1]
    output_category: used for predicting the category, (dims: [batch, n_classes]
    y: the class label
    '''
    #step 1: loss for real_fake
    batch_size = output_real_fake.size(0)
    pt_real_fake = output_real_fake
    logpt_real_fake = -torch.log(pt_real_fake)
    if real_fake_weight is None:
        real_fake_weight = pt_real_fake.detach()
        p = pt_real_fake*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p * logpt_real_fake
    else:
        real_fake_weight = torch.cat([real_fake_weight, pt_real_fake], 1)
        p = torch.mean(real_fake_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p*logpt_real_fake
    loss_real_fake = torch.mean(loss_real_fake)
    
    #Step 2: output_category has been sigmoid
    y = y.view(y.size(0), 1)
    pt_category = (1.-y.float())*(1-output_category) + y.float()*output_category
    logpt_category = -torch.log(pt_category)
    
    if category_weight is None:
        category_weight = pt_category.detach()
        p = pt_category*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        category_weight = torch.cat([category_weight, pt_category], 1)
        p = torch.mean(category_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_category = p*logpt_category
    loss_category = torch.mean(logpt_category)

    return loss_real_fake, loss_category, real_fake_weight, category_weight

def loss_for_fakeV2(output_real_fake, output_category, y, real_fake_weight=None, 
                    category_weight=None, gamma=1.0, weights=None):
    '''
    output_real_fake: the output used for predicting whether the input is real or fake(dims:[batch, 1], range in [0,1]
    output_category: used for predicting the category, (dims: [batch, n_classes]
    y: the class label
    '''
    #step 1: loss for real_fake
    batch_size = output_real_fake.size(0)
    pt_real_fake = 1.0 - output_real_fake
    logpt_real_fake = -torch.log(pt_real_fake)
    if real_fake_weight is None:
        real_fake_weight = pt_real_fake.detach()
        p = pt_real_fake*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p * logpt_real_fake
    else:
        real_fake_weight = torch.cat([real_fake_weight, pt_real_fake], 1)
        p = torch.mean(real_fake_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
        loss_real_fake = p*logpt_real_fake
    loss_real_fake = torch.mean(loss_real_fake)

    #Step 2: loss for category
    y = y.view(y.size(0), 1)
    pt_category = (1.-y.float())*(1-output_category) + y.float()*output_category
    logpt_category = -torch.log(pt_category)
    
    if category_weight is None:
        category_weight = pt_category.detach()
        p = pt_category*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        category_weight = torch.cat([category_weight, pt_category], 1)
        p = torch.mean(category_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_category = p*logpt_category
    loss_category = torch.mean(logpt_category)

    return loss_real_fake, loss_category, real_fake_weight, category_weight

def CSV_data_Loading(path):
    # loading data
    df = pd.read_csv(path) 
    
    labels = df['class']
    
    x_df = df.drop(['class'], axis=1)
    
    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)
    x = np.array(x)
    labels = np.array(labels)
    
    return x, labels;

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ensemble.")
    parser.add_argument('--data_name', nargs='?', default='Annthyroid',
                        help='Input data name.')
    parser.add_argument('--max_epochs', type=int, default=11,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--lr_g', type=float, default=0.01,
                        help='Learning rate of generator.')
    parser.add_argument('--lr_d', type=float, default=0.01,
                        help='Learning rate of discriminator.')
    parser.add_argument('--active_rate', type=float, default=0.05,
                        help='the proportion of instances that need to be labeled.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size.')
    parser.add_argument('--dim_z', type=int, default=128,
                        help='dim for latent noise.')
    parser.add_argument('--dis_layer', type=int, default=1,
                        help='hidden_layer number in dis.')
    parser.add_argument('--gen_layer', type=int, default=2,
                        help='hidden_layer number in gen.')
    parser.add_argument('--ensemble_num', type=int, default=10,
                        help='the number of dis in ensemble.')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='if GPU used')
    parser.add_argument('--SN_used', type=bool, default=True,
                        help='if spectral Normalization used')
    parser.add_argument('--init_type', nargs='?', default="ortho",
                        help='init method for both gen and dis, including ortho,N02,xavier')
    parser.add_argument('--print', type=bool, default=True,
                        help='Print the learning procedure')
    return parser.parse_args()

def data_norm(dataset):
    minVals = dataset.min(0)
    maxVals = dataset.max(0)
    ranges = maxVals-minVals
    norm_data = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    norm_data = dataset - np.tile(minVals, (m,1))
    norm_data = norm_data / np.tile(ranges, (m, 1))
    return norm_data

