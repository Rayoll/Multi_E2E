import os
import argparse
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from model.DBNet import DBNet_MultiTask
from utils.dataset_ import POIDataset,AOI_POI_Dataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils.process import dataSplit,dbMulti_evaluate
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs',type=int,default=4)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--epoch',type=int,default=15)
    parser.add_argument('--lyr',type=int,default=2)

    args = parser.parse_args()

    return args

def AOIPOItrain(model,train_loader,val_loader,device,args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    accs, kappas = [], []
    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        count = 0
        with tqdm(total=args.ntrain, desc=f'epoch{epoch + 1}/{args.epoch}') as pbar:
            for batch, (image, aoi_labels, batched_graph, graph_labels, graphid) in enumerate(train_loader):
                image = image.to(device)
                aoi_labels = aoi_labels.to(device)
                batched_graph = batched_graph.to(device)
                feats = batched_graph.ndata.pop('attr')
                logits, f_socio, f_vis = model(batched_graph, feats, image)
                loss = criterion(logits, aoi_labels) + criterion(f_socio,aoi_labels) + criterion(f_vis,aoi_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pbar.update(len(image))

                count += 1
            scheduler.step()

        train_acc, train_kappa, _ = dbMulti_evaluate(train_loader, device, model, criterion)
        val_acc, val_kappa, val_loss = dbMulti_evaluate(val_loader, device, model, criterion)
        print(
            "Epoch {:05d} | Loss {:.3f} | ValLoss {:.3f} | Train Acc. {:.4f} | Train Kappa. {:.4f} | Val Acc. {:.4f}| Val Kappa. {:.4f}  ".format(
                epoch, total_loss / count, val_loss, train_acc, train_kappa, val_acc, val_kappa
            )
        )
        logging.info(
            "Epoch {:05d} | Loss {:.3f} | ValLoss {:.3f} | Train Acc. {:.4f} | Train Kappa. {:.4f} | Val Acc. {:.4f}| Val Kappa. {:.4f}  ".format(
                epoch, total_loss / count, val_loss, train_acc, train_kappa, val_acc, val_kappa
            )
        )

        accs.append(val_acc)
        kappas.append(val_kappa)

    accs, kappas = np.array(accs), np.array(kappas)
    accs.sort()
    kappas.sort()
    accs = accs[::-1]
    kappas = kappas[::-1]
    accs, kappas = accs[:3], kappas[:3]

    return accs, kappas

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False # tmp
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s',filename='train_log.txt')

    gt = pd.read_csv(r'./data/example_label_gt.txt',dtype={'pid':str})
    gt_type = gt['gt_1800'].values
    graphType = pd.read_csv('./data/graph_type_15.txt', header=None).values
    tmp = graphType[:, 0].copy()
    graphType[:, 0] = graphType[:, 1]
    graphType[:, 1] = tmp
    graphType = dict(graphType)
    aoi_Dir = './data/example_scene_block/'
    AOI_ids = os.listdir(aoi_Dir)
    AOI_ids.sort(key=lambda x: int(x[:-4]))
    AOI_ids = [x[:-4] for x in AOI_ids]
    AOI_ids = np.array(AOI_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    poi_dataset = POIDataset(raw_dir='./data/example_DT_GraphData.txt',
                        save_dir='./cache',
                        cityName="example")
    in_channels = poi_dataset.dim_nfeats
    out_channels = poi_dataset.gclasses
    valid_AOI_withPOIs_id = np.loadtxt(
        f'./data/graph_index_data/example_DT_GraphIndexData.txt',
        dtype=str)

    valid_AOI_withPOIs_id = valid_AOI_withPOIs_id[1:]
    AOI_withPOI_id = valid_AOI_withPOIs_id
    AOI_withPOI_label = np.zeros(len(AOI_withPOI_id), dtype=np.int8)

    for idx, aoi_id in enumerate(AOI_withPOI_id):
        a = gt[gt['pid'] == aoi_id]
        b = a['gt_1800'].values
        label = graphType[b[0]]
        AOI_withPOI_label[idx] = label

    train_idx, val_idx = dataSplit(aoi_labels=AOI_withPOI_label, train_ratio=0.7, seed=3407)

    args.ntrain = len(train_idx)
    args.nval = len(val_idx)

    aoi_poi_dataset = AOI_POI_Dataset(filepath=aoi_Dir,
                                      imgs_dir=valid_AOI_withPOIs_id,
                                      labels=AOI_withPOI_label,
                                      transform=transforms.Compose([transforms.Resize(224),transforms.ToTensor()]),
                                      raw_dir='./data/example_DT_GraphData.txt',
                                      save_dir='./cache',
                                      cityName='example',
                                      )


    aoi_poi_train_loader = GraphDataLoader(
        aoi_poi_dataset,
        sampler=SubsetRandomSampler(np.array(train_idx)),
        batch_size=args.bs,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    aoi_poi_val_loader = GraphDataLoader(
        aoi_poi_dataset,
        sampler=SubsetRandomSampler(np.array(val_idx)),
        batch_size=args.bs,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    model = DBNet_MultiTask(hid_channels=256,out_channels=15,args=args,r=16).to(device)
    accs, kappas = AOIPOItrain(model,aoi_poi_train_loader,aoi_poi_val_loader,device,args)

    logging.info('************************************')
    logging.info(
        'oa={:.4f}, kappa={:.4f}'.format(np.mean(accs),np.mean(kappas)))
    logging.info('************************************')
    print('************************************')
    print(
        'oa={:.4f}, kappa={:.4f}'.format(np.mean(accs), np.mean(kappas)))
    print('************************************')

