import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F

def loss_dis_real(dis_real, out_category, y, weights=None, cat_weight=None, gamma=2.0):
    #step 1: the loss for GAN
    logpt = F.softplus(-dis_real)
    pt = torch.exp(-logpt)
    if weights is None:
        weights = pt.detach()
        p = pt*1
        #print(dis_real.shape)
        #print(p.shape)
        p = p.view(dis_real.shape[0], 1)
        p = (1-p)**gamma
        loss = p.clone().detach() * logpt
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(len(dis_real), 1)
        p = (1-p)**gamma
        loss = p.clone().detach()*logpt
    loss = torch.mean(loss)

    #step 2: loss for classifying
    target = y.view(y.size(0), 1)
    pt_cat = (1.-target.float())*(1-out_category) + target.float()*out_category
    logpt_cat = -torch.log(pt_cat)
    batch_size = target.size(0)
    
    if cat_weight is None:
        cat_weight = pt_cat.detach()
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = torch.cat([cat_weight, pt_cat], 1)
        p = torch.mean(cat_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p.clone().detach()*logpt_cat
    loss_cat = torch.mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight


def loss_dis_fake(dis_fake, out_category, y, weights=None, cat_weight=None, gamma=2.0):
    logpt = F.softplus(dis_fake)
    pt = torch.exp(-logpt)

    if weights is None:
        p = pt*1
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p.clone().detach() * logpt
        weights = pt.detach()
        #loss = logpt
    else:
        weights = torch.cat([weights, pt], 1)
        p = torch.mean(weights, 1)
        p = p.view(len(dis_fake), 1)
        p = (1-p)**gamma
        loss = p.clone().detach()*logpt
    
    loss = torch.mean(loss)

    #step 2: loss for classifying
    target = y.view(y.size(0), 1)
    pt_cat = (1.-target.float())*(1-out_category) + target.float()*out_category
    logpt_cat = -torch.log(pt_cat)
    batch_size = target.size(0)
    
    if cat_weight is None:
        cat_weight = pt_cat.detach()
        p = pt_cat*1
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    else:
        cat_weight = torch.cat([cat_weight, pt_cat], 1)
        p = torch.mean(cat_weight, 1)
        p = p.view(batch_size, 1)
        p = (1-p)**gamma
    logpt_cat = p.clone().detach()*logpt_cat
    loss_cat = torch.mean(logpt_cat)
    return loss, loss_cat, weights, cat_weight

