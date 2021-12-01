from collections import defaultdict
import functools
from numpy.random import gamma
import torch
import torch.nn as nn
from torch.optim import Adam
#import torch.optim.lr_scheduler.StepLR as StepLR

import numpy as np
import pandas as pd
from tqdm import tqdm


from src.BigGANdeep import Generator, Discriminator
from src.my_utils import active_sampling, AUC_and_Gmean
from src.loss import loss_dis_fake, loss_dis_real
import src.utils as utils

class EAL_GAN(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.device = 'cuda' if args.cuda else 'cpu'

        self.generator = Generator(G_ch=args.channel, dim_z=args.dim_z, resolution=args.resolution, n_classes=2, output_channel=args.feat_dim)
        self.optimizer_g = Adam(self.generator.parameters(), lr=args.lr_g, betas=(0.00, 0.99))
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(self.optimizer_g, step_size=args.step_size, gamma=args.gamma)
        if args.cuda:
            self.generator = nn.DataParallel(self.generator).cuda()

        self.NetD_Ensemble = []
        self.Opti_Ensemble = []
        self.schD_Ensemble = []
        lr_ds = np.random.rand(args.ensemble_num)*(args.lr_d*5-args.lr_d)+args.lr_d  #learning rate
        for index in range(args.ensemble_num):
            netD = Discriminator(D_ch=args.channel, resolution=args.resolution, n_classes=2, input_channel=args.feat_dim)
            optimizerD = Adam(netD.parameters(), lr=args.lr_d, betas=(0.00, 0.99))
            schedule_D = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=args.step_size, gamma=args.gamma)
            if args.cuda:
                netD = nn.DataParallel(netD).cuda()

            self.NetD_Ensemble += [netD]
            self.Opti_Ensemble += [optimizerD]
            self.schD_Ensemble += [schedule_D]
    
    def fit(self):
        self.iterations = 0
        
        self.z, self.y = utils.prepare_z_y(self.args.batch_size, dim_z=self.args.dim_z, nclasses=2, device=self.device)
        z_for_sample, y_for_sample = utils.prepare_z_y(self.args.batch_size, dim_z=self.args.dim_z, nclasses=2, device=self.device)
        self.sample = functools.partial(utils.sample, G=self.generator, z_=z_for_sample, y_=y_for_sample)
        self.train_history = defaultdict(list)

        Best_Measure_Recorded = 0
        for epoch in range(self.args.max_epochs):
            
            auc_train, gmean_train, auc_test, gmean_test = self.train_one_epoch(epoch)
            if auc_train*gmean_train > Best_Measure_Recorded:
                Best_Measure_Recorded = auc_train*gmean_train
                Best_AUC = auc_test
                Best_Gmean = gmean_test

                states = {
                    'epoch':epoch,
                    'gen_dict':self.generator.state_dict(),
                    'auc_train':auc_train,
                    'auc_test': auc_test
                }
                for i in range(self.args.ensemble_num):
                    netD = self.NetD_Ensemble[i]
                    states['dis_dict'+str(i)] = netD.state_dict()
                    
                torch.save(states, './logs/checkpoint_best.pth')
            #print(train_AUC, test_AUC, epoch)
            print('Training for epoch %d: Train_AUC=%.4f train_Gmean=%.4f Test_AUC=%.4f  Test_Gmean=%.4f' % (epoch + 1, auc_train, gmean_train, auc_test, gmean_test))
        
        
        '''
        #step 1: load the best models
        self.Best_Ensemble = []
        states = torch.load('./logs/checkpoint_best.pth')
        self.generator.load_state_dict(states['gen_dict'])
        for i in range(self.args.ensemble_num):
            netD = self.NetD_Ensemble[i]
            netD.load_state_dict(states['dis_dict'+str(i)])
            self.Best_Ensemble += [netD]
        '''
        
        return Best_AUC, Best_Gmean
    

    def train_one_epoch(self, epoch=1):
        train_loader, test_loader = self.dataset.loaders(batch_size=self.args.batch_size, num_workers=0)
        for (inputs, targets, semi_argets,_) in tqdm(train_loader):
            #print("inputs_shape=", inputs.shape)
            self.iterations += 1
            #step 1: update the discriminators
            if self.args.cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            self.z.sample_()
            self.y.sample_()
            generated_x = self.generator(self.z, self.y)
            generated_x = generated_x.detach()

            #print("generated_shape=", generated_x.shape)
            
            real_weights = None
            fake_weights = None
            real_cat_weights = None
            fake_cat_weights = None
            
            dis_loss = 0
            gen_loss = 0
            #select p% of the training data, label them
            real_x_selected, real_y_selected, _, index_selected = active_sampling(self.args, inputs, targets, self.NetD_Ensemble, need_sample=(self.iterations>1))
            
            for i in range(self.args.ensemble_num):
                optimizer = self.Opti_Ensemble[i]
                netD = self.NetD_Ensemble[i]
                out_real_fake, out_real_categoy = netD(real_x_selected, real_y_selected)
        
                loss_adv_real, real_loss_cat, real_weights, real_cat_weights = loss_dis_real(out_real_fake, out_real_categoy, real_y_selected, real_weights, real_cat_weights)
                real_loss = loss_adv_real+real_loss_cat
                
                #train on fake data
                output, out_fake_category = netD(generated_x, self.y.detach())
                loss_adv_fake, loss_cat_fake, fake_weights, fake_cat_weights = loss_dis_fake(output, out_fake_category, self.y, fake_weights, fake_cat_weights)
                fake_loss = loss_adv_fake + loss_cat_fake
                sum_loss = real_loss+fake_loss
                dis_loss += sum_loss

                self.train_history['discriminator_loss_'+str(i)].append(sum_loss)
                optimizer.zero_grad()
                sum_loss.backward()
                optimizer.step()
                #print('real_loss=', real_loss.data, "    fake_loss=", fake_loss.data)
            self.train_history['discriminator_loss'].append(dis_loss)

            #train the generator
            self.z.sample_()
            self.y.sample_()
            generated_x = self.generator(self.z, self.y)
            
            gen_loss = 0
            gen_weights = None
            gen_cat_weights = None
            for i in range(self.args.ensemble_num):
                #optimizer = names['optimizerD_' + str(i)]
                netD = self.NetD_Ensemble[i]
                output, out_category = netD(generated_x, self.y)
                #out_real_fake, out_real_categoy = netD(real_x, real_y)
                loss, loss_cat, gen_weights, gen_cat_weights = loss_dis_real(output, out_category, self.y, gen_weights, gen_cat_weights)
                #loss, gen_weights = loss_gen(output, gen_weights)
                gen_loss += (loss+loss_cat)
            self.train_history['generator_loss'].append(gen_loss)
            self.optimizer_g.zero_grad()
            gen_loss.backward()
            self.optimizer_g.step()

            if self.iterations%10==0:
                print("dis_loss=%.4f   gen_loss=%.4f" % (gen_loss.data, dis_loss.data ))
        
        torch.cuda.empty_cache()

        y_train, y_scores = self.predict(train_loader, self.NetD_Ensemble)
        #print("anomaly number=", np.sum(y_train))
        y_scores_pandas = pd.DataFrame(y_scores)
        y_scores_pandas.fillna(0, inplace=True)
        y_scores = y_scores_pandas.values

        auc, gmean = AUC_and_Gmean(y_train, y_scores)
        self.train_history['train_auc'].append(auc)
        self.train_history['train_Gmean'].append(gmean)

        y_test, y_scores_test = self.predict(test_loader, self.NetD_Ensemble)
        y_scores_pandas = pd.DataFrame(y_scores_test)
        y_scores_pandas.fillna(0, inplace=True)
        y_scores_test = y_scores_pandas.values

        test_auc, test_gmean = AUC_and_Gmean(y_test, y_scores_test)

        return auc, gmean, test_auc, test_gmean
    
    def predict(self, data_loader, dis_Ensemble=None):
        p = []
        y = []
        for (real_x, real_y, _, index) in tqdm(data_loader):
            final_pt = 0
            for i in range(self.args.ensemble_num):
                pt = self.Best_Ensemble[i](real_x, mode=1) if dis_Ensemble is None else dis_Ensemble[i](real_x, mode=1)
                final_pt = pt.cpu().detach().numpy() if i==0 else final_pt+pt.cpu().detach().numpy()

            final_pt /= self.args.ensemble_num
            #final_pt = final_pt.view(-1,)
            
            p += [final_pt]
            y += [real_y.cpu().detach().numpy()]
        #p = torch.cat(p, 0).cpu().detach().numpy()
        #y = torch.cat(y, 0).cpu().detach().numpy()
        #print(p)
        p = np.concatenate(p, 0)
        y = np.concatenate(y, 0)
        return y, p


            

    
