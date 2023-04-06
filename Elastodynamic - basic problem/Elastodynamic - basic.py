
import numpy as np
import time
from pyDOE import lhs
import matplotlib
import platform
if platform.system()=='Linux':
    matplotlib.use('Agg')
if platform.system()=='Windows':
    from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import pickle
import math
import scipy.io
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import mean_squared_error
#### Creating distance Modelwith Pytorch
# torch.cuda.empty_cache()

class ANN_DistModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=20, N_LAYERS=4):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


#### Creating part Modelwith Pytorch

class ANN_PartModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=20, N_LAYERS=4):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


#### Creating Modelwith Pytorch

class ANN_UvModel(nn.Module):
    def __init__(self, N_INPUT=2, N_OUTPUT=5, N_HIDDEN=70, N_LAYERS=8):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(N_INPUT, N_HIDDEN),
                        activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(N_HIDDEN, N_HIDDEN),
                            activation()]) for _ in range(N_LAYERS-1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)
        self.apply(self._init_weights)
    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_normal_(module.weight)
          if module.bias is not None:
              module.bias.data.zero_()
                
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def GenDistPt(xmin, xmax, ymin, ymax, xc, yc, r, num_surf_pt, num):
    # num: number per edge
    # num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    x, y = np.meshgrid(x, y)

    x = x.flatten()[:, None]
    y = y.flatten()[:, None]

    return x,y

def GenDist(XY_dist):
    dist_u = np.zeros_like(XY_dist[:, 0:1])
    dist_v = np.zeros_like(XY_dist[:, 0:1])
    dist_s11 = np.zeros_like(XY_dist[:, 0:1])
    dist_s22 = np.zeros_like(XY_dist[:, 0:1])
    dist_s12 = np.zeros_like(XY_dist[:, 0:1])
    for i in range(len(XY_dist)):
        dist_u[i, 0] = XY_dist[i][0]  # min(t, x-(-0.5))
        dist_v[i, 0] =  XY_dist[i][0]  # min(t, sqrt((x+0.5)^2+(y+0.5)^2))
        dist_s11[i, 0] = 0.5 - XY_dist[i][0]
        dist_s22[i, 0] = min(0.5 - XY_dist[i][1],XY_dist[i][1])
        dist_s12[i, 0] = min(XY_dist[i][1], 0.5 - XY_dist[i][1], 0.5 - XY_dist[i][0])
    DIST = np.concatenate(( dist_u, dist_v, dist_s11, dist_s22, dist_s12), 1)
    return XY_dist , DIST


 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Current cuda device: ',torch.cuda.get_device_name(0))


E = 20.0
mu = 0.25
rho = 1.0

PI = math.pi
MAX_T = 10.0

# Domain bounds for x, y and t
lb = np.array([0, 0])
ub = np.array([0.5, 0.5])

# Network configuration
uv_layers   = [2] + 8 * [70] + [5]
dist_layers = [2] + 4 * [20] + [5]
part_layers = [2] + 4 * [20] + [5]

# Generate distance function for spatio-temporal space
x_dist, y_dist = GenDistPt(xmin=0, xmax=0.5, ymin=0, ymax=0.5, xc=0, yc=0, r=0.1,
                                    num_surf_pt=40, num=100)
XY_dist = np.concatenate((x_dist, y_dist), 1)
XY_dist,DIST = GenDist(XY_dist)

# Collocation point for equation residual
XY_c = lb + (ub - lb) * lhs(2, 30000)

LW = np.array([0.0, 0.0]) + np.array([0.5, 0.0]) * lhs(2, 5000)
UP = np.array([0.0, 0.5]) + np.array([0.5, 0.0]) * lhs(2, 5000)
LF = np.array([0.0, 0.0]) + np.array([0.0, 0.5]) * lhs(2, 5000)
RT = np.array([0.5, 0.0]) + np.array([0.0, 0.5]) * lhs(2, 8000)

# t_RT = RT[:, 2:3]
# period = 5  # two period in 10s
# s11_RT = 0.5 * np.sin((2 * PI / period) * t_RT + 3 * PI / 2) + 0.5
s11_RT=np.ones(RT[:,0:1].shape)
RT = np.concatenate((RT, s11_RT), 1)

# Add some boundary points into the collocation point set
XY_c = np.concatenate((XY_c, LF[::5, :], RT[::5, 0:2], UP[::5, :], LW[::5, :]), 0)

XY_dist=torch.FloatTensor(XY_dist).requires_grad_(True)
DIST=torch.FloatTensor(DIST).requires_grad_(True)
LW=torch.FloatTensor(LW).requires_grad_(True)
LF=torch.FloatTensor(LF).requires_grad_(True)
UP=torch.FloatTensor(UP).requires_grad_(True)
RT=torch.FloatTensor(RT).requires_grad_(True)
XY=torch.FloatTensor(XY_c).requires_grad_(True)


# train standard neural network to fit distance training data

torch.manual_seed(123)
model_dist = ANN_DistModel()
if os.path.isfile('weights/distance.pth'):
    model_dist.load_state_dict(torch.load('weights/distance.pth'))
    print("loaded distance model successfully...")
model_dist.to(device)


optimizer_dist = torch.optim.Adam(model_dist.parameters(),lr=1e-3)
for i in range(1000):
    optimizer_dist.zero_grad()
    XY_dist=XY_dist.to(device)
    DIST=DIST.to(device)
    # compute the "data loss"
    yh = model_dist(XY_dist)

    loss1 = torch.mean((yh-DIST)**2)# use mean squared error
  
    
    # backpropagate joint loss
    # loss=loss1
    loss = 1000*(loss1)# add two loss terms together
    if i%100==0: 
      print("Epoch number: {} and the loss : {}".format(i,loss.item()))

    loss.backward()
    optimizer_dist.step()


# train standard neural network to fit distance training data
torch.save(model_dist.state_dict(), "weights/distance.pth")
print("saved distance model successfully...")

torch.manual_seed(123)
model_part = ANN_PartModel()
if os.path.isfile('weights/part.pth'):
    model_part.load_state_dict(torch.load('weights/part.pth'))
    print("loaded part model successfully...")
model_part.to(device)

optimizer_part = torch.optim.Adam(model_part.parameters(),lr=1e-3)
for i in range(1000):
    optimizer_part.zero_grad()

    #LF
    LF=LF.to(device)
    yh_lf = model_part(LF)
    loss1=torch.mean((yh_lf[:,0])**2)+torch.mean((yh_lf[:,1])**2)


    #UP
    UP=UP.to(device)
    yh_up = model_part(UP)
    loss1=loss1+torch.mean((yh_up[:,3])**2)+torch.mean((yh_up[:,4])**2);
    #LW
    LW=LW.to(device);
    yh_lw = model_part(LW)
    loss1=loss1+torch.mean((yh_lw[:,3])**2)+torch.mean((yh_lw[:,4])**2);

    #RT
    RT=RT.to(device);
    yh_rt = model_part(RT[:,0:2])
    loss1=loss1+torch.mean((yh_rt[:,4])**2);
    loss1=loss1+torch.mean((yh_rt[:,2]-RT[:,2])**2);
    
    # backpropagate joint loss
    # loss=loss1
    loss = 1000*(loss1)# add two loss terms together
    if i%100==0: 
      print("Epoch number: {} and the loss : {}".format(i,loss.item()))

    loss.backward()
    optimizer_part.step()


torch.save(model_part.state_dict(), "weights/part.pth")
print("saved part model successfully...")
# train standard neural network to fit distance training data
torch.manual_seed(123)
model_uv = ANN_UvModel()
if os.path.isfile('weights/uv_original.pth'):
    model_uv.load_state_dict(torch.load('weights/uv_original.pth'))
    print("loaded uv model successfully...")
model_uv.to(device)


train_loader = torch.utils.data.DataLoader(XY, batch_size=32)
optimizer_uv = torch.optim.Adam(model_uv.parameters(),lr=1e-4)
for i in range(5):
    running_loss=0.0 
    for _, xy in enumerate(train_loader):
                
            optimizer_uv.zero_grad()
            # compute the "data loss"
            xy=xy.to(device)
            yh = model_uv(xy)
            yh=yh*model_dist(xy)+model_part(xy)

            e11=torch.autograd.grad(yh[:,0], xy,torch.ones_like(yh[:,0]), create_graph=True)[0][:,0]# computes dy/dx
            e12=torch.autograd.grad(yh[:,0], xy,torch.ones_like(yh[:,0]), create_graph=True)[0][:,1]
            e22=torch.autograd.grad(yh[:,1], xy,torch.ones_like(yh[:,1]), create_graph=True)[0][:,1]# computes dy/dx
            e12=e12+torch.autograd.grad(yh[:,1], xy,torch.ones_like(yh[:,1]), create_graph=True)[0][:,0]
            sp11 = (E * e11)/ (1 - mu * mu) + (E * mu * e22) / (1 - mu * mu)
            sp22 = (E * mu* e11) / (1 - mu * mu)  + (E * e22) / (1 - mu * mu) 
            sp12 = (E* e12) / (2 * (1 + mu)) 

            f_s11 = yh[:,2] - sp11
            f_s12 = yh[:,4] - sp12
            f_s22 = yh[:,3] - sp22

            s11_1=torch.autograd.grad(yh[:,2], xy,torch.ones_like(yh[:,2]), create_graph=True)[0][:,0]
            s12_2=torch.autograd.grad(yh[:,4], xy,torch.ones_like(yh[:,4]), create_graph=True)[0][:,1]

            s22_2=torch.autograd.grad(yh[:,3], xy,torch.ones_like(yh[:,3]), create_graph=True)[0][:,1]
            s12_1=torch.autograd.grad(yh[:,4], xy,torch.ones_like(yh[:,4]), create_graph=True)[0][:,0]



            f_u = s11_1 + s12_2 
            f_v = s22_2 + s12_1 
            # print("f_s11",torch.mean(f_s11**2),"f_s22",torch.mean(f_s22**2),"f_s12",f_s12,"f_u",f_u,"f_v",f_v)
            loss1 = torch.mean(f_s11**2)+torch.mean(f_s22**2)+torch.mean(f_s12**2)+torch.mean(f_u**2)+torch.mean(f_v**2)# use mean squared error

            loss = loss1 # add two loss terms together 
            loss=10*loss

            loss.backward()
            optimizer_uv.step()
            # print statistics
            running_loss += loss.item()
            if _ % 100 == 99:    # print every 200 mini-batches
                torch.save(model_uv.state_dict(), "weights/uv_original.pth")
                print('[%d, %5d] loss: %.4f' %
                    (i + 1, _ + 1, running_loss / 100))
                running_loss = 0.0


def postProcessDef(xmin, xmax, ymin, ymax, field, s=5, num=0, scale=1):
    ''' Plot deformed plate (set scale=0 want to plot undeformed contours)
    '''
    # [x_pred, y_pred, u_pred, v_pred, s11_pred, s22_pred, s12_pred] = field
    x_pred=field[:,0]
    y_pred=field[:,1]
    u_pred=field[:,2]
    v_pred=field[:,3]
    s11_pred=field[:,4]
    s22_pred=field[:,5]
    s12_pred=field[:,6]
    # print(v_pred)
    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=u_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title(r'$u$-error', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)
    #
    cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=v_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    for key, spine in ax[1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1].axis('square')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title(r'$v$-error', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)
    #
    plt.savefig('./output/uv_comparison' + str(num) + '.png', dpi=200)
    plt.close('all')
    #
    # Plot predicted stress
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
    fig.subplots_adjust(hspace=0.15, wspace=0.3)
    #
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s11_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title(r'$\sigma_{11}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)
    
    #
    cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s22_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[1].axis('square')
    for key, spine in ax[1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title(r'$\sigma_{22}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)
    #
    cf = ax[2].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=s12_pred, alpha=0.7, edgecolors='none',
                          marker='s', cmap='rainbow', s=s)
    ax[2].axis('square')
    for key, spine in ax[2].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[2].set_title(r'$\sigma_{12}$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2])
    cbar.ax.tick_params(labelsize=14)
    #
    plt.savefig('./output/stress_comparison' + str(num) + '.png', dpi=200)
    plt.close('all')

def FEMcomparisionUV(xyt, y_predicted, uv_fem,  s=5):
    x_pred=xyt[:,0]
    y_pred=xyt[:,1]
    u_pred=y_predicted[:,0]
    v_pred=y_predicted[:,1]
    u_fem=uv_fem[:,0]
    v_fem=uv_fem[:,1]
    u_error=u_pred-u_fem
    v_error=v_pred-v_fem
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.suptitle(f"t=0")


    cf= ax[0,0].scatter(x_pred , y_pred, c=u_pred, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.025)
    ax[0, 0].set_title(r'$u$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    
    cf=ax[1,0].scatter(x_pred, y_pred, c=u_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.025)
    ax[1, 0].set_title(r'$u$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 0])
    
    cf=ax[2,0].scatter(x_pred, y_pred, c=np.abs(u_error), alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.025)
    
    ax[2, 0].set_title(r'$u$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 0])

    cf= ax[0,1].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=-0.004,vmax=0.004)
    ax[0, 1].set_title(r'$v$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    
    cf=ax[1,1].scatter(x_pred, y_pred, c=v_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=-0.004,vmax=0.004)
    ax[1, 1].set_title(r'$v$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 1])
    
    cf=ax[2,1].scatter(x_pred, y_pred, c=np.abs(v_error), alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.004)
    
    ax[2, 1].set_title(r'$v$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 1])
    plt.savefig('./output/uv_error_'+f"t=0"+ '.png', dpi=200)
    plt.close('all')    

def FEMcomparisionStr(xyt, s_predicted, s_fem,  s=5):
    x_pred=xyt[:,0]
    y_pred=xyt[:,1]
    s11_pred=s_predicted[:,2]
    s12_pred=s_predicted[:,4]
    s22_pred=s_predicted[:,3]
    s11_fem=s_fem[:,0]
    s12_fem=s_fem[:,1]
    s22_fem=s_fem[:,2]
    s11_error=s11_pred-s11_fem
    s12_error=s12_pred-s12_fem
    s22_error=s22_pred-s22_fem

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig.tight_layout(pad=4.0)
    fig.suptitle(f"t=0")


    lblsz=8

    cf= ax[0,0].scatter(x_pred , y_pred , c=s11_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s,vmin=0.95,vmax=2.25)
    ax[0,0].axis('square')
    ax[0, 0].set_title(r'$s11$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 0])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,0].scatter(x_pred, y_pred, c=s11_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=0.95,vmax=2.25)
    ax[1,0].axis('square')
    ax[1, 0].set_title(r'$s11$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 0])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,0].scatter(x_pred , y_pred, c=np.abs(s11_error), alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=0,vmax=2.25)
    ax[2,0].axis('square')
    
    ax[2, 0].set_title(r'$s11$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 0])
    cbar.ax.tick_params(labelsize=lblsz)


    cf= ax[0,1].scatter(x_pred , y_pred , c=s12_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s,vmin=-0.4,vmax=0.4)
    ax[0,1].axis('square')
    ax[0, 1].set_title(r'$s12$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 1])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,1].scatter(x_pred, y_pred, c=s12_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=-0.4,vmax=0.4)
    ax[1,1].axis('square')
    ax[1, 1].set_title(r'$s12$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 1])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,1].scatter(x_pred , y_pred, c=np.abs(s12_error), alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.4)
    ax[2,1].axis('square')
    
    ax[2, 1].set_title(r'$s12$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 1])
    cbar.ax.tick_params(labelsize=lblsz)



    cf= ax[0,2].scatter(x_pred , y_pred , c=s22_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.45)
    ax[0,2].axis('square')
    ax[0, 2].set_title(r'$s22$-PINN', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0, 2])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[1,2].scatter(x_pred, y_pred, c=s22_fem, alpha=0.7, edgecolors='none',
                        cmap='rainbow', marker='o', s=s,vmin=0,vmax=0.45)
    ax[1,2].axis('square')
    ax[1, 2].set_title(r'$s22$-FEM', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1, 2])
    cbar.ax.tick_params(labelsize=lblsz)
    
    cf=ax[2,2].scatter(x_pred , y_pred, c=np.abs(s22_error), alpha=0.7, edgecolors='none',
                    cmap='rainbow', marker='o',vmin=0,vmax=0.45)
    ax[2,2].axis('square')
    ax[2, 2].set_title(r'$s22$-error', fontsize=8)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[2, 2])
    cbar.ax.tick_params(labelsize=lblsz)

    plt.savefig('./output/stress_error_'+f"t=0"+ '.png', dpi=200)
    plt.close('all')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# x_star = np.linspace(0, 0.5, 251)
# y_star = np.linspace(0, 0.5, 251)
# x_star, y_star = np.meshgrid(x_star, y_star)
# x_star = x_star.flatten()[:, None]
# y_star = y_star.flatten()[:, None]
shutil.rmtree('./output', ignore_errors=True)
if not os.path.isdir('./output'):
    os.makedirs('./output')

# os.makedirs('./output')
path_fem = './platehinge_data.csv'

my_data = np.genfromtxt(path_fem, delimiter=',')
csv_data=my_data[1:,:]
x_col, y_col = 2,3
u_col, v_col = 0,1
s11_col, s12_col, s22_col=5,6,8
xyt=torch.FloatTensor(csv_data[:,[x_col,y_col]]).requires_grad_(True)
xyt=xyt.to(device)
uv_fem=torch.FloatTensor(csv_data[:,[u_col,v_col]]).requires_grad_(True)
s_fem=torch.FloatTensor(csv_data[:,[s11_col,s12_col,s22_col]]).requires_grad_(True)

y=model_uv(xyt)
yd=model_dist(xyt)
yp=model_part(xyt)
y=y*yd+yp
y=y.cpu().detach().numpy()
xyt=xyt.cpu().detach().numpy()
uv_fem=uv_fem.cpu().detach().numpy()
s_fem=s_fem.cpu().detach().numpy()
rmse_u=mean_squared_error(uv_fem[:,0],y[:,0],squared=False)
rmse_v=mean_squared_error(uv_fem[:,1],y[:,1],squared=False)
rmse_s11=mean_squared_error(s_fem[:,0],y[:,2],squared=False)
rmse_s22=mean_squared_error(s_fem[:,2],y[:,3],squared=False)
rmse_s12=mean_squared_error(s_fem[:,1],y[:,4],squared=False)

FEMcomparisionUV(xyt,y,uv_fem)
FEMcomparisionStr(xyt,y,s_fem)


# xy=np.concatenate((x_star,y_star),1)
# xy=torch.FloatTensor(xy).requires_grad_(True)
# xy=xy.to(device) 
# y=model_uv(xy)
# yd=model_dist(xy)
# yp=model_part(xy)
# y=y*yd+yp


# xy=xy.cpu().detach().numpy()
# y=y.cpu().detach().numpy()



# field=np.concatenate((xy,y),1)

# postProcessDef(xmin=0, xmax=0.50, ymin=0, ymax=0.50, s=4, scale=0, field=field)