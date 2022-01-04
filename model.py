#%%
import torch.nn as nn
import torch
import torch.functional as F
from utils.pyart import *


class POELayer(nn.Module):
    def __init__(self, branchLs):
        super(POELayer, self).__init__()

        self.branchLS = branchLs
        n_joint = len(branchLs)

        self.twist = nn.Parameter(torch.Tensor(n_joint,6))
        self.twist.data.uniform_(-1,1)

        for joint in range(n_joint):
            setattr(self,'branch'+str(joint)+'_p',nn.Parameter(torch.Tensor(1,3).uniform_(-1,1)) )
            setattr(self,'branch'+str(joint)+'_rpy',nn.Parameter(torch.Tensor(1,3).uniform_(-1,1)) )
        

    def forward(self, q_value):
        branchLs = self.branchLS
        n_joint = len(branchLs)
        batch_size = q_value.size()[0]
        device = q_value.device
        out = torch.tile(torch.eye(4),(batch_size,1,1)).to(device)
        Twistls = torch.zeros([batch_size,n_joint,6]).to(device)

        outs = torch.tensor([]).reshape(batch_size,-1,4,4).to(device)
        for joint in range(n_joint):
            twist = self.twist[joint,:]
            Twistls[:,joint,:] = inv_x(t2x(out))@twist
            out = out @ srodrigues(twist, q_value[:,joint])

            if branchLs[joint]:
                p = getattr(self,'branch'+str(joint)+'_p')
                rpy = getattr(self,'branch'+str(joint)+'_rpy')
                r = rpy2r(rpy)
                out_temp = out @ pr2t(p, r)
                outs = torch.cat((outs,out_temp.unsqueeze(1)), dim=1)

        return outs,Twistls

class q_layer(nn.Module):
    def __init__(self,branchLs,inputdim,n_layers=7):
        super(q_layer, self).__init__()
        n_joint = len(branchLs)
        
        LayerList = []
        for _ in range(n_layers):
            layer = nn.Linear(inputdim,2*inputdim)
            torch.nn.init.xavier_uniform_(layer.weight)
            LayerList.append(layer)
            inputdim = inputdim * 2

        for _ in range(n_layers-3):
            layer = nn.Linear(inputdim,inputdim//2)
            torch.nn.init.xavier_uniform_(layer.weight)
            LayerList.append(layer)
            inputdim = inputdim // 2

        layer = nn.Linear(inputdim,n_joint)
        torch.nn.init.xavier_uniform_(layer.weight)
        LayerList.append(layer)

        self.layers = torch.nn.ModuleList(LayerList)
        

    def forward(self, motor_control):
        out =motor_control
        
        for layer in self.layers:
            out = layer(out)
            out = torch.nn.LeakyReLU()(out)
    
        q_value = out
        return q_value

class Model(nn.Module):
    def __init__(self, branchLs, inputdim):
        super(Model,self).__init__()
        self.q_layer = q_layer(branchLs, inputdim)
        self.poe_layer = POELayer(branchLs)

    def forward(self, motor_control):
        out = self.q_layer(motor_control)
        SE3,_ = self.poe_layer(out)

        return SE3




#%%