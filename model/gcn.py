import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv,GINConv, SAGEConv
from dgl.nn.pytorch.glob import MaxPooling
import dgl



# Graph Convolutional Network
class WA_GCN(nn.Module):
    def __init__(self,hid_channels,out_channels,num_classes,init_weight_fn,args):
        super().__init__()
        self.num_layers = args.lyr
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.linear_prediction = nn.ModuleList()

        self.weights = torch.from_numpy(np.load(init_weight_fn,allow_pickle=True)).float().to(torch.device('cuda'))


        for layer in range(self.num_layers):
            if layer == 0:
                self.layers.append(GraphConv(num_classes,hid_channels))
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))
            else:
                self.layers.append(GraphConv(hid_channels,hid_channels))
                self.batch_norms.append(nn.BatchNorm1d(hid_channels))

        self.fc = nn.Linear(hid_channels,out_channels)
        self.drop = nn.Dropout(p=0.5)
        self.pooling = (MaxPooling())

    def forward(self,g,h):
        h = torch.matmul(h,self.weights.transpose(1,0))
        for i, layer in enumerate(self.layers):
            h = layer(g,h)
            h = self.batch_norms[i](h)
            h = F.relu(h)

        with g.local_scope():
            g.ndata['h'] = h
            h = dgl.max_nodes(g,'h')
            h = self.fc(h)

        return h

















