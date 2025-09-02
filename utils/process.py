import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


# 评价指标计算
def calOA(mtx):
    OA = np.sum(np.diag(mtx))/np.sum(mtx)
    return OA
def calKappa(mtx):
    p0 = calOA(mtx)
    pe = np.sum(np.sum(mtx,axis=1)*np.diag(mtx))/(np.sum(mtx)**2)
    kappa = (p0 - pe)/(1 - pe)
    return kappa

def dbMulti_evaluate(dataloader,device,model,criterion,filename=None):
    model.eval()
    total_loss, total = 0., 0
    confusionMtx = torch.zeros((15,15))
    for batch, (image, aoi_labels, batched_graph, _, graphid) in enumerate(dataloader):
        image = image.to(device)
        aoi_labels = aoi_labels.to(device)
        batched_graph = batched_graph.to(device)
        feats = batched_graph.ndata.pop('attr')
        # output
        logits, f_socio, f_vis = model(batched_graph, feats, image)
        _, predicted = torch.max(logits, dim=1)
        loss = criterion(logits, aoi_labels) + criterion(f_socio,aoi_labels) + criterion(f_vis,aoi_labels)
        total_loss += loss.item()
        total += 1
        # calculate confusion matrix
        for i, pred in enumerate(predicted):
            confusionMtx[aoi_labels[i].cpu().data, pred.cpu().data] += 1

    confusionMtx = np.array(confusionMtx)
    total_loss /= total
    kappa = calKappa(confusionMtx)
    acc = calOA(confusionMtx)
    if filename is not None:
        np.savetxt(filename,confusionMtx)

    return acc, kappa, total_loss


# 数据划分，每次划分的时候都打乱顺序
def dataSplit(aoi_labels,train_ratio,seed):
    aoi_labels = np.array(aoi_labels)
    train_idx, val_idx = [], []
    aoi_classes_count = np.bincount(aoi_labels)
    np.random.seed(seed)
    for aoi_label, num in enumerate(aoi_classes_count):
        aoi_label_index = np.where(aoi_labels==aoi_label)[0]
        # 打乱顺序
        index = np.random.permutation(aoi_label_index.size)
        aoi_label_index = aoi_label_index[index]
        train_idx = train_idx + aoi_label_index[:int(train_ratio*num)].tolist()

        val_idx = val_idx + aoi_label_index[int(train_ratio*num):].tolist()

    return train_idx, val_idx

