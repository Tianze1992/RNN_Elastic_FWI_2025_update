"""
This class offers the objective functions used for FWI 
- obj_option = 1: L1 norm is used 
- obj_option = 2: L2 norm is used  
- obj_option = 3: Correlatoin based objective function is used 
- obj_option = 4: Zero mean correlatoin based objective function is used 
@author: Tianze Zhang
- By Tianze Zhang on June. 14, 2022
- University of calgary
"""
import numpy as np
import torch
import math

class FWI_costfunction(torch.nn.Module):
    def __init__(self, obj_option):
        super(FWI_costfunction, self).__init__()
        self.obj_option = obj_option
        if self.obj_option == 1:
            print("Using the L1 Norm objective function")
        if self.obj_option == 2:
            print("Using the L2 Norm objective function")
        if self.obj_option == 3:
            print("Using the Correlation based objective function")
        if self.obj_option == 4:
            print("Using the Zero Mean Correlation based objective function")
        
    
    def L2_norm(self,shot_syn, shot_obs):
        Costfunction = torch.nn.MSELoss(reduction='sum')
        loss_Seg = Costfunction(shot_syn,shot_obs)
        return loss_Seg
    
    
    def L1_norm(self,shot_syn, shot_obs):
        Costfunction = torch.nn.L1Loss(reduction='sum')
        loss_Seg = Costfunction(shot_syn,shot_obs)
        return loss_Seg
    
    
    def global_correlation_misfit(self, shot_syn, shot_obs):
        [num_batch, num_shot] = [shot_syn.shape[0],shot_syn.shape[1]]
        loss_Seg = 0
        for ibatch in range(num_batch):
            for ishot in range (num_shot):
                i_shot_true =  shot_obs[ibatch, ishot,:,:]
                i_shot_pred =  shot_syn[ibatch, ishot,:,:]
                correlation_shot_true_pred = i_shot_true*i_shot_pred
                correlation_shot_true_true = i_shot_true*i_shot_true
                correlation_shot_pred_pred = i_shot_pred*i_shot_pred
                E_true = torch.sum(correlation_shot_true_true)
                E_pred = torch.sum(correlation_shot_pred_pred)
                global_correlation_res_shot = correlation_shot_true_pred/(torch.sqrt(E_true)*torch.sqrt(E_pred))
                loss_Seg = loss_Seg + torch.sum(global_correlation_res_shot)*(-1)
        return loss_Seg
    
    
    def zero_mean_global_correlation_misfit(self, shot_syn,shot_obs):
        [num_batch, num_shot] = [shot_syn.shape[0],shot_syn.shape[1]]
        loss_Seg = 0
        for ibatch in range(num_batch):
          for ishot in range (num_shot):
              i_shot_true = shot_obs[ibatch, ishot,:,:]
              i_shot_pred = shot_syn[ibatch, ishot,:,:]

              mean_shot_true_pred = torch.mean(i_shot_pred)
              mean_shot_true_true = torch.mean(i_shot_true)
              
              correlation_shot_true_pred = (i_shot_true - mean_shot_true_true)*(i_shot_pred - mean_shot_true_pred)
              correlation_shot_true_true = (i_shot_true - mean_shot_true_true)*(i_shot_true - mean_shot_true_true)
              correlation_shot_pred_pred = (i_shot_pred - mean_shot_true_pred)*(i_shot_pred - mean_shot_true_pred)
              
              E_true = torch.sum(correlation_shot_true_true)
              E_pred = torch.sum(correlation_shot_pred_pred)

              global_correlation_res_shot = correlation_shot_true_pred/(torch.sqrt(E_true)*torch.sqrt(E_pred))
              loss_Seg = loss_Seg + torch.sum(global_correlation_res_shot)*(-1)
        
        return loss_Seg
    
    
    def forward(self, shot_syn, shot_obs):
        if self.obj_option == 2:
            loss_Seg = self.L2_norm(shot_syn, shot_obs)
        if self.obj_option == 1:
            loss_Seg = self.L1_norm(shot_syn, shot_obs)
        if self.obj_option == 3:
            loss_Seg = self.global_correlation_misfit(shot_syn, shot_obs)
        if self.obj_option == 4:
            loss_Seg = self.zero_mean_global_correlation_misfit(shot_syn, shot_obs)
        return loss_Seg

