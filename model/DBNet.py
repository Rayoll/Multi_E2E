import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import ResNet, Bottleneck
from model.gcn import WA_GCN


class DBNet_MultiTask(nn.Module):
    def __init__(self,hid_channels,out_channels,args,r=16):
        super().__init__()
        self.gcn = WA_GCN(hid_channels,hid_channels//2,out_channels,'../ckpt/poi_lu_init_weight.npy',args)
        self.resnet = ResNet(Bottleneck,[3,4,6,3])
        self.resnet.load_state_dict(torch.load('../ckpt/resnet50-19c8e357.pth'))
        self.resnet.fc = nn.Linear(2048,hid_channels//2)

        self.resSE = nn.Sequential(
            nn.Linear(hid_channels//2,hid_channels//(2*r)),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels//(2*r),hid_channels//2),
            nn.Sigmoid(),
        )
        self.gcnSE = nn.Sequential(
            nn.Linear(hid_channels//2,hid_channels//(2*r)),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels//(2*r),hid_channels//2),
            nn.Sigmoid(),
        )
        self.gcn_fc = nn.Linear(hid_channels//2,out_channels)
        self.res_fc = nn.Linear(hid_channels//2,out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hid_channels,out_channels)

    def forward(self,g,h,x):
        gcn_feats = self.gcn(g,h)
        res_feats = self.resnet(x)

        gcn_feats_opt = self.gcn_fc(gcn_feats)
        res_feats_opt = self.res_fc(res_feats)

        # 自适应融合权值
        gcn_weight = self.gcnSE(gcn_feats)
        res_weight = self.resSE(res_feats)
        gcn_feats = res_weight * gcn_feats
        res_feats = gcn_weight * res_feats

        combined_feats = torch.concat([gcn_feats,res_feats],dim=1)

        output = self.fc(combined_feats)

        return output, gcn_feats_opt, res_feats_opt

