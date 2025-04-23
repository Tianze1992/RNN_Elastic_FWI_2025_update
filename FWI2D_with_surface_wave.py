"""
@author: Jian Sun
- By Jian Sun on Sep. 20, 2020
- Ocean univeristy of China
@author: Tianze Zhang
- By Tianze Zhang on June. 14, 2022
- University of calgary
"""
import torch
import torch.utils.data

from rnn_fd_elastic2_free_surface import rnn2D
from generator import gen_Segment2d
from RNN_FWI_objective_function import FWI_costfunction


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return [repackage_hidden(v) for v in h]

#############################################################################################
# ##                           Full Waveform Inversion  Model                             ###
#############################################################################################
class FWI2D():
    def __init__(self, 
                 segment_size,
                 ns=1, 
                 nz=None,
                 nx=None,
                 zs=None,
                 xs=None,
                 zr=None, 
                 xr=None,
                 dz=None,
                 dt=None,
                 nt = None,
                 npad=0, 
                 order=2, 
                 vmax=6000,
                 vpadding=None,
                 obj_option = 2,
                 freeSurface=True,
                 dtype=torch.float32,
                 device='cpu'):
        super(FWI2D, self).__init__()

        self.segment_size = segment_size
        self.ns = ns

        self.obj_option = obj_option
        self.nz = nz
        self.nx = nx
        self.zs = zs
        self.xs = xs
        self.zr = zr
        self.xr = xr
        self.dz = dz
        self.dt = dt
        self.nt = nt 
        self.npad = npad
        self.order = order
        self.freeSurface = freeSurface
        self.vmax = vmax
        self.dtype = dtype
        self.device = device

        self.nx_pad = nx + 2 * npad
        self.nz_pad = nz + npad if freeSurface else nz + 2 * npad
        
    
        self.Costfunction = FWI_costfunction(self.obj_option)
        self.rnn = rnn2D(self.nz, self.nx, self.zs, self.xs, self.zr, self.xr, self.dz, self.dt, self.npad, self.order, self.vmax, self.freeSurface, self.dtype, self.device).to(self.device)

    def train(self, 
              MaxIter, 
              vmodel1,
              vmodel2,
              vmodel3,
              wavelet=None, 
              shots=None, 
              option=0, 
              log_interval=1, 
              resume_file_name=None):

        vmodel1 = torch.as_tensor(vmodel1).requires_grad_(True).to(self.device)
        vmodel2 = torch.as_tensor(vmodel2).requires_grad_(True).to(self.device)      
        vmodel3 = torch.as_tensor(vmodel3).requires_grad_(True).to(self.device)

        optimizer1 = torch.optim.Adam([vmodel1],lr=10/2)
        optimizer2 = torch.optim.Adam([vmodel2],lr=5/2)
        optimizer3 = torch.optim.Adam([vmodel3],lr=1e-1)
        #print("if the velocity models are leaf points ======>", vmodel1.is_leaf,vmodel2.is_leaf,vmodel3.is_leaf)
        
        best_loss = 1e100
        best_loss_epoch = 0
        resume_from_epoch = 0
        train_loss_history = []
        if isinstance(resume_file_name, str):
            resume_from_epoch, best_loss, best_loss_epoch, best_loss_model, \
                train_loss_history, optimizer = self.load_state(resume_file_name, optimizer)
        #vmodel = torch.cat((vmodel1, vmodel2, vmodel3),dim=0)
        for epoch in range(resume_from_epoch, MaxIter):
            loss, segment_ytPred_x,segment_ytPred_z = self.train_one_epoch(None, optimizer1, optimizer2, optimizer3,vmodel1,vmodel2,vmodel3, wavelet, shots, option)
            train_loss_history.append(loss.item())
            if epoch % log_interval == 0 or epoch == MaxIter - 1:
                print("Epoch: {:5d}, Loss: {:.4e}".format(epoch, loss.item()))
                torch.save(train_loss_history,'./training_loss_history.pt')
            
        return train_loss_history, vmodel1, vmodel2, vmodel3, segment_ytPred_x,segment_ytPred_z

    def train_one_epoch(self, optimizer0, optimizer1, optimizer2, optimizer3, vmodel1, vmodel2, vmodel3 , wavelet=None, shots=None, option=0):
        shots = shots.to(self.device)

        loss = 0
        for iseg, (segWavelet, segData) in enumerate(gen_Segment2d(wavelet, shots, segment_size=self.segment_size, option=option)):
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            #print("if the velocity models are leaf points ======>", vmodel1.is_leaf,vmodel2.is_leaf,vmodel3.is_leaf)
            vx_save, vz_save, txx_save, tzz_save, txz_save, \
            segment_ytPred_x,segment_ytPred_z,\
            avg_regularizer, velocity_output1,velocity_output2,velocity_output3= self.forward_process(vmodel1, vmodel2, vmodel3, segWavelet, option) 

            vx_save = repackage_hidden(vx_save)
            vz_save = repackage_hidden(vz_save)
            txx_save = repackage_hidden(txx_save)
            tzz_save = repackage_hidden(tzz_save)
            txz_save = repackage_hidden(txz_save)

            shots_pred= torch.cat((segment_ytPred_x.reshape(1, self.ns, self.nt , len(self.xr)), segment_ytPred_z.reshape(1, self.ns, self.nt, len(self.xr))),dim=0)
            
            loss_Seg = self.Costfunction(shots_pred,segData)
            loss_Seg.backward()

            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            loss += loss_Seg.detach()
        return loss.cpu().detach(), segment_ytPred_x,segment_ytPred_z
            
    def forward_process(self, vmodel1, vmodel2, vmodel3, wavelet=None,option=0):
        vx_save, vz_save, txx_save, tzz_save, txz_save, \
        segment_ytPred_x,segment_ytPred_z,\
        avg_regularizer, vmodel1_out, vmodel2_out, vmodel3_out = self.rnn(vmodel1, vmodel2, vmodel3, wavelet, option)

        return vx_save, vz_save, txx_save, tzz_save, txz_save, \
               segment_ytPred_x,segment_ytPred_z,\
               avg_regularizer, vmodel1_out, vmodel2_out, vmodel3_out