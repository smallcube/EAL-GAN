import numpy as np
import math
import functools
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.font_manager

from numpy import percentile
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P

from .layers import ccbn, identity, SNLinear, SNEmbedding

from .utils import prepare_z_y, active_sampling_V1, sample_selector_V1
from .losses import loss_dis_fake, loss_dis_real
from .pyod_utils import AUC_and_Gmean


class Generator(nn.Module):
    def __init__(self, dim_z=64, hidden_dim=128, output_dim=128, n_classes=2, hidden_number=1,
                 init='ortho', SN_used=True):
        super(Generator, self).__init__()
        self.hidden_dim = hidden_dim

        self.n_classes = n_classes
        self.init = init
        self.shared_dim = dim_z // 2

        if SN_used:
            self.which_linear = functools.partial(SNLinear,
                                                  num_svs=1, num_itrs=1)
            # self.which_embedding = functools.partial(SNEmbedding,
            #                        num_svs=1, num_itrs=1)
        else:
            self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding

        self.shared = self.which_embedding(n_classes, self.shared_dim)
        self.input_fc = self.which_linear(dim_z + self.shared_dim, self.hidden_dim)

        self.output_fc = self.which_linear(self.hidden_dim, output_dim)

        self.model = nn.Sequential(self.input_fc,
                                   nn.ReLU())

        for index in range(hidden_number):
            middle_fc = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.model.add_module('hidden-layers-{0}'.format(index), middle_fc)
            self.model.add_module('ReLu-{0}'.format(index), nn.ReLU())
            # self.model.add_module('ReLu-{0}'.format(index), nn.Tanh())

        self.model.add_module('output_layer', self.output_fc)

        self.init_weights()

    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for G''s initialized parameters: %d' % self.param_count)

    def forward(self, z, y):
        # If hierarchical, concatenate zs and ys
        y = self.shared(y)  # modification
        h = torch.cat([z, y], 1)

        h = self.model(h)
        # Apply batchnorm-relu-conv-tanh at output
        return h


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=1, n_classes=2,
                 hidden_number=1, init='ortho', SN_used=True):
        super(Discriminator, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.init = init

        if SN_used:
            self.which_linear = functools.partial(SNLinear,
                                                  num_svs=1, num_itrs=1)
            # self.which_embedding = functools.partial(SNEmbedding,
            #                        num_svs=1, num_itrs=1)
        else:
            self.which_linear = nn.Linear
        self.which_embedding = nn.Embedding

        self.input_fc = self.which_linear(input_dim, self.hidden_dim)
        self.output_fc = self.which_linear(self.hidden_dim, output_dim)
        self.output_category = nn.Sequential(self.which_linear(self.hidden_dim, output_dim),
                                             nn.Sigmoid())
        # Embedding for projection discrimination
        self.embed = self.which_embedding(self.n_classes, self.hidden_dim)
        self.model = nn.Sequential(self.input_fc,
                                   nn.ReLU())

        # self.blocks = []
        for index in range(hidden_number):
            middle_fc = self.which_linear(self.hidden_dim, self.hidden_dim)
            self.model.add_module('hidden-layers-{0}'.format(index), middle_fc)
            self.model.add_module('ReLu-{0}'.format(index), nn.ReLU())
            # self.model.add_module('ReLu-{0}'.format(index), nn.Sigmoid())

        self.init_weights()

    # Initialize
    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        # print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self, x, y=None, mode=0):
        # mode 0: train the whole discriminator network
        if mode == 0:
            h = self.model(x)
            out = self.output_fc(h)
            # Get projection of final featureset onto class vectors and add to evidence
            out_real_fake = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
            out_category = self.output_category(h)
            return out_real_fake, out_category
        # mode 1: train self.output_fc, only classify whether an input is fake or real
        elif mode == 1:
            h = self.model(x)
            out = self.output_fc(h)
            return out
        # mode 2: train self.output_category, used in fine_tunning stage
        else:
            h = self.model(x)
            out = self.output_category(h)
            return out


class EAL_GAN(nn.Module):
    def __init__(self, args, data_x, data_y, test_x, test_y, visualize=False):
        super(EAL_GAN, self).__init__()

        lr_g = args.lr_g
        lr_d = args.lr_d

        self.device = torch.device("cuda" if args.cuda else "cpu")
        z, y = prepare_z_y(20, args.dim_z, 2, device=self.device)
        y = y * 0 + 1
        self.y = y.long()
        self.noise = z

        self.args = args
        # self.data_x = torch.from_numpy(data_x).float()
        # self.data_y = torch.from_numpy(data_y).long()
        self.data_x = data_x
        self.data_y = data_y
        self.test_x = test_x
        self.test_y = test_y
        self.iterations = 0
        self.visualize = visualize

        self.feature_size = data_x.shape[1]
        self.data_size = data_x.shape[0]
        self.batch_size = min(args.batch_size, self.data_size)
        self.hidden_dim = self.feature_size * 2
        self.dim_z = args.dim_z

        manualSeed = random.randint(1, 10000)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # 1: prepare Generator
        self.netG = Generator(dim_z=self.dim_z, hidden_dim=self.hidden_dim, output_dim=self.feature_size, n_classes=2,
                              hidden_number=args.gen_layer, init=args.init_type, SN_used=args.SN_used)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr_g, betas=(0.00, 0.99))

        if args.cuda:
            self.netG = nn.DataParallel(self.netG, device_ids=[0, 1])
            self.netG = self.netG.to(self.device)

        # 2: create ensemble of discriminator
        self.NetD_Ensemble = []
        self.opti_Ensemble = []
        lr_ds = np.random.rand(args.ensemble_num) * (args.lr_d * 5 - args.lr_d) + args.lr_d  # learning rate
        for index in range(args.ensemble_num):
            netD = Discriminator(input_dim=self.feature_size, hidden_dim=self.hidden_dim, output_dim=1, n_classes=2,
                                 hidden_number=args.dis_layer, init=args.init_type, SN_used=args.SN_used)
            optimizerD = optim.Adam(netD.parameters(), lr=lr_ds[index], betas=(0.00, 0.99))
            if args.cuda:
                netD = nn.DataParallel(netD, device_ids=[0, 1])
                netD = netD.to(self.device)

            self.NetD_Ensemble += [netD]
            self.opti_Ensemble += [optimizerD]

    def fit(self):
        log_dir = os.path.join('./log/', self.args.data_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        z, y = prepare_z_y(self.batch_size, self.dim_z, 2, device=self.device)
        # Start iteration
        Best_Measure_Recorded = -1
        best_auc = 0
        best_gmean = 0
        self.train_history = defaultdict(list)
        for epoch in range(self.args.max_epochs):
            train_AUC, train_Gmean, test_auc, test_gmean = self.train_one_epoch(z, y, epoch)
            if train_Gmean * train_AUC > Best_Measure_Recorded:
                Best_Measure_Recorded = train_Gmean * train_AUC
                best_auc = test_auc
                best_gmean = test_gmean
                states = {
                    'epoch': epoch,
                    'gen_dict': self.netG.state_dict(),
                    'opti_gen': self.optimizerG.state_dict(),
                    'max_auc': train_AUC
                }
                for i in range(self.args.ensemble_num):
                    netD = self.NetD_Ensemble[i]
                    optimi_D = self.opti_Ensemble[i]
                    states['dis_dict' + str(i)] = netD.state_dict()
                    states['opti_dis' + str(i)] = optimi_D.state_dict()

                torch.save(states, os.path.join(log_dir, 'checkpoint_best.pth'))
            # print(train_AUC, test_AUC, epoch)
            if self.args.print:
                print('Training for epoch %d: Train_AUC=%.4f train_Gmean=%.4f Test_AUC=%.4f  Test_Gmean=%.4f' % (
                epoch + 1, train_AUC, train_Gmean, test_auc, test_gmean))

        # step 1: load the best models
        self.Best_Ensemble = []
        states = torch.load(os.path.join(log_dir, 'checkpoint_best.pth'))
        self.netG.load_state_dict(states['gen_dict'])
        for i in range(self.args.ensemble_num):
            netD = self.NetD_Ensemble[i]
            netD.load_state_dict(states['dis_dict' + str(i)])
            self.Best_Ensemble += [netD]

        return best_auc, best_gmean

    def predict(self, test_x, test_y, dis_Ensemble=None):
        p = []
        y = []

        data_size = test_x.shape[0]
        num_batches = data_size // self.batch_size
        num_batches = num_batches + 1 if data_size % self.batch_size > 0 else num_batches

        for index in range(num_batches):
            end_pos = min(data_size, (index + 1) * self.batch_size)
            real_x = test_x[index * self.batch_size: end_pos]
            real_y = test_y[index * self.batch_size: end_pos]

            final_pt = 0
            for i in range(self.args.ensemble_num):
                pt = self.Best_Ensemble[i](real_x, mode=2) if dis_Ensemble is None else dis_Ensemble[i](real_x, mode=2)
                final_pt = pt.detach() if i == 0 else final_pt + pt

            final_pt /= self.args.ensemble_num
            final_pt = final_pt.view(-1, )

            p += [final_pt]
            y += [real_y]
        p = torch.cat(p, 0).cpu().detach().numpy()
        y = torch.cat(y, 0).cpu().detach().numpy()
        return y, p

    def train_one_epoch(self, z, y, epoch=1):
        # train discriminator & generator for one specific spoch

        data_size = self.data_x.shape[0]
        # feature_size = data_x.shape[1]
        batch_size = min(self.args.batch_size, data_size)

        num_batches = data_size // batch_size
        num_batches = num_batches + 1 if data_size % batch_size > 0 else num_batches

        # data shuffer
        perm_index = torch.randperm(data_size)
        if self.args.cuda:
            perm_index = perm_index.cuda()
        # data_x = data_x[perm_index]
        # data_y = data_y[perm_index]

        x_empirical, y_empirical = [], []


        for index in range(num_batches):
            # step 1: train the ensemble of discriminator
            # Get training data
            self.iterations += 1

            end_pos = min(data_size, (index + 1) * batch_size)
            real_x = self.data_x[index * batch_size: end_pos]
            real_y = self.data_y[index * batch_size: end_pos]

            real_weights = None
            fake_weights = None
            real_cat_weights = None
            fake_cat_weights = None

            z.sample_()
            y.sample_()
            generated_x = self.netG(z, y)
            generated_x = generated_x.detach()

            losses = []
            dis_loss = 0
            gen_loss = 0
            # select p% of the training data, label them
            real_x_selected, real_y_selected, _ = active_sampling_V1(self.args, real_x, real_y, self.NetD_Ensemble,
                                                                     need_sample=(self.iterations > 1))
            x_empirical += [real_x_selected]
            y_empirical += [real_y_selected]
            x_empirical += [generated_x]
            y_empirical += [y]
            # print(real_y_selected.shape)
            for i in range(self.args.ensemble_num):
                optimizer = self.opti_Ensemble[i]
                netD = self.NetD_Ensemble[i]
                # train the GAN with real data

                out_real_fake, out_real_categoy = netD(real_x_selected, real_y_selected)

                loss1, real_loss_cat, real_weights, real_cat_weights = loss_dis_real(out_real_fake, out_real_categoy,
                                                                                     real_y_selected, real_weights,
                                                                                     real_cat_weights)
                real_loss = loss1 + real_loss_cat

                # train on fake data
                output, out_fake_category = netD(generated_x, y.detach())
                loss2, loss_cat_fake, fake_weights, fake_cat_weights = loss_dis_fake(output, out_fake_category, y,
                                                                                     fake_weights, fake_cat_weights)
                fake_loss = loss2 + loss_cat_fake
                sum_loss = real_loss + fake_loss
                dis_loss += sum_loss

                self.train_history['discriminator_loss_' + str(i)].append(sum_loss)
                losses += [sum_loss]
                optimizer.zero_grad()
                sum_loss.backward(retain_graph=True)
                optimizer.step()
            self.train_history['discriminator_loss'].append(dis_loss)

            # step 2: train the generator
            z.sample_()
            y.sample_()
            generated_x = self.netG(z, y)
            gen_loss = 0
            gen_weights = None
            gen_cat_weights = None
            for i in range(self.args.ensemble_num):
                # optimizer = names['optimizerD_' + str(i)]
                netD = self.NetD_Ensemble[i]
                output, out_category = netD(generated_x, y)
                # out_real_fake, out_real_categoy = netD(real_x, real_y)
                loss, loss_cat, gen_weights, gen_cat_weights = loss_dis_real(output, out_category, y, gen_weights,
                                                                             gen_cat_weights)
                # loss, gen_weights = loss_gen(output, gen_weights)
                gen_loss += (loss + loss_cat)
            self.train_history['generator_loss'].append(gen_loss)
            self.optimizerG.zero_grad()
            gen_loss.backward()
            self.optimizerG.step()

        x_empirical = torch.cat(x_empirical, 0)
        y_empirical = torch.cat(y_empirical, 0)
        y_train, y_pred_train = self.predict(x_empirical, y_empirical, self.NetD_Ensemble)

        y_scores_pandas = pd.DataFrame(y_pred_train)
        y_scores_pandas.fillna(0, inplace=True)
        y_pred_train = y_scores_pandas.values

        auc, gmean = AUC_and_Gmean(y_train, y_pred_train)
        self.train_history['train_auc'].append(auc)
        self.train_history['train_Gmean'].append(gmean)

        y_test, y_pred_test = self.predict(self.test_x, self.test_y, self.NetD_Ensemble)
        y_scores_pandas = pd.DataFrame(y_pred_test)
        y_scores_pandas.fillna(0, inplace=True)
        y_pred_test = y_scores_pandas.values
        test_auc, test_gmean = AUC_and_Gmean(y_test, y_pred_test)

        return auc, gmean, test_auc, test_gmean

