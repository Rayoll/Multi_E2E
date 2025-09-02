import os
import torch
import numpy as np
from dgl import backend as F
from dgl.convert import graph as dgl_graph
from dgl.data import DGLDataset
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from PIL import Image



'''数据加载模块'''

# custom dataset
# getitem,process,len是必需的
class POIDataset(DGLDataset):
    def __init__(self,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 cityName=None):

        self.cityName = cityName

        self.graphs = [] # graphs
        self.labels = [] # graphs label
        self.graphID = [] # graphs id

        # 图/节点/边数统计
        self.N = 0  # graph num
        self.n = 0  # node num
        self.m = 0  # edge num

        # label dictionary
        self.glabel_dict = {}
        self.nlabel_dict = {}
        # super
        super().__init__(name='dataset_name',
                         url=url,
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return g, self.labels[idx], self.graphID[idx]

    def process(self):
        if self.verbose:
            print('loading ckpt...')
        self.file = self.raw_dir
        with open(self.file,mode='r',encoding='utf-8') as f:
            # graph num
            self.N = int(f.readline().strip())
            for i in range(self.N):
                if (i + 1) % 10 == 0 and self.verbose:
                    print('processing graph {}...'.format(i + 1))
                n_nodes, glabel, graph_id = f.readline().strip().split() # graph info
                n_nodes, glabel = int(n_nodes), int(glabel)

                self.graphID.append(graph_id)


                self.labels.append(glabel)
                # create dgl graph
                g = dgl_graph(([],[]))
                g.add_nodes(n_nodes)

                # node labels
                nlabels = []
                # nattrs = [] # save node attribute --> POI id
                m_edges = 0

                for j in range(n_nodes):
                    nrow = f.readline().strip().split()
                    nrow = [int(w) for w in nrow]

                    if not nrow[0] in self.nlabel_dict:
                        self.nlabel_dict[nrow[0]] = nrow[0]

                    nlabels.append(nrow[0])

                    m_edges += nrow[1]
                    g.add_edges(j,nrow[2:])

                    # add self loop
                    m_edges += 1
                    g.add_edges(j,j)
                    # info print
                    if (j+1)%10 == 0 and self.verbose is True:
                        print(
                            'processing node {} of graph {}...'.format(
                                j+1,i+1
                            ))
                        print('this node has {} edges'.format(nrow[1]))

                g.ndata['label'] =F.tensor(nlabels)

                # update statistics of graph
                self.n += n_nodes
                self.m += m_edges

                self.graphs.append(g)
        # 所有的self.labels转换为one-hot向量
        # self.labels: 128维 list
        for i in range(15):
           self.glabel_dict[i] = i

        self.labels = F.tensor(self.labels)
        # generate one-hot attribute for node according to its label
        nlabel_set = set([])
        for g in self.graphs:
            nlabel_set = nlabel_set.union(
                set([F.as_scalar(nl) for nl in g.ndata['label']])
            )
        nlabel_set = list(nlabel_set)
        label2idx = {
            nlabel_set[i]: i
            for i in range(len(nlabel_set))
        }
        for g in self.graphs:
            attr = np.zeros((
                g.number_of_nodes(),len(nlabel_set)
            ))
            attr[range(g.number_of_nodes()),[label2idx[nl] for nl in F.asnumpy(g.ndata['label']).tolist()]] = 1
            g.ndata['attr'] = F.tensor(attr, F.float32)

        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.nlabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata['attr'][0])

        if self.verbose:
            print('Done')
            print(
                """
                -------- Data Statistics --------'
                #Graphs: %d
                #Graph Classes: %d
                #Nodes: %d
                #Node Classes: %d
                #Node Features Dim: %d
                #Edges: %d
                Avg. of #Nodes: %.2f
                Avg. of #Edges: %.2f \n """ % (
                    self.N, self.gclasses, self.n, self.nclasses,
                    self.dim_nfeats, self.m,
                    self.n / self.N, self.m / self.N))

    def save(self):
        graph_path = os.path.join(self.save_dir,f'{self.cityName}_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        label_dict = {'labels': self.labels}
        info_dict = {'N': self.N,
                     'n': self.n,
                     'm': self.m,
                     'gclasses': self.gclasses,
                     'nclasses': self.nclasses,
                     'dim_nfeats': self.dim_nfeats,
                     'glabel_dict': self.glabel_dict,
                     'nlabel_dict': self.nlabel_dict,
                     'graph_ids':self.graphID}

        save_graphs(str(graph_path),self.graphs,label_dict)
        save_info(str(info_path),info_dict)
        pass

    def load(self):
        graph_path = os.path.join(self.save_dir,f'{self.cityName}_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graphs = graphs
        self.labels = label_dict['labels']

        self.N = info_dict['N']
        self.n = info_dict['n']
        self.m = info_dict['m']
        self.gclasses = info_dict['gclasses']
        self.nclasses = info_dict['nclasses']
        self.dim_nfeats = info_dict['dim_nfeats']
        self.glabel_dict = info_dict['glabel_dict']
        self.nlabel_dict = info_dict['nlabel_dict']
        self.graphID = info_dict['graph_ids']

    def has_cache(self):
        graph_path = os.path.join(self.save_dir,f'{self.cityName}_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    @property
    def num_classes(self):
        return self.gclasses

class AOI_POI_Dataset(DGLDataset):
    def __init__(self,
                 filepath,
                 imgs_dir,
                 labels,
                 transform=None,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 cityName=None):

        # AOI dataset
        self.filepath = filepath
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform

        # POI dataset
        self.cityName = cityName

        self.graphs = [] # graphs
        self.labels = [] # graphs label
        self.graphID = [] # graphs id

        # 图/节点/边数统计
        self.N = 0  # graph num
        self.n = 0  # node num
        self.m = 0  # edge num

        # label dictionary
        self.glabel_dict = {}
        self.nlabel_dict = {}
        # super
        super().__init__(name='dataset_name',
                         url=url,
                         raw_dir=raw_dir,
                         save_dir=save_dir,
                         force_reload=force_reload,
                         verbose=verbose)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # image ckpt
        img_path = os.path.join(self.filepath,self.imgs_dir[idx]+'.png')
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # graph ckpt
        g = self.graphs[idx]

        return img,self.labels[idx],\
               g, self.labels[idx], self.graphID[idx]

    def process(self):
        if self.verbose:
            print('loading ckpt...')
        self.file = self.raw_dir
        with open(self.file,mode='r',encoding='utf-8') as f:
            # graph num
            self.N = int(f.readline().strip())
            for i in range(self.N):
                if (i + 1) % 10 == 0 and self.verbose:
                    print('processing graph {}...'.format(i + 1))
                n_nodes, glabel, graph_id = f.readline().strip().split() # graph info
                n_nodes, glabel = int(n_nodes), int(glabel)

                self.graphID.append(graph_id)


                self.labels.append(glabel)
                # create dgl graph
                g = dgl_graph(([],[]))
                g.add_nodes(n_nodes)

                # node labels
                nlabels = []
                # nattrs = [] # save node attribute --> POI id
                m_edges = 0

                for j in range(n_nodes):
                    nrow = f.readline().strip().split()
                    nrow = [int(w) for w in nrow]

                    if not nrow[0] in self.nlabel_dict:
                        self.nlabel_dict[nrow[0]] = nrow[0]

                    nlabels.append(nrow[0])

                    m_edges += nrow[1]
                    g.add_edges(j,nrow[2:])

                    # add self loop
                    m_edges += 1
                    g.add_edges(j,j)
                    # info print
                    if (j+1)%10 == 0 and self.verbose is True:
                        print(
                            'processing node {} of graph {}...'.format(
                                j+1,i+1
                            ))
                        print('this node has {} edges'.format(nrow[1]))

                g.ndata['label'] =F.tensor(nlabels)

                # update statistics of graph
                self.n += n_nodes
                self.m += m_edges

                self.graphs.append(g)
        # 所有的self.labels转换为one-hot向量
        # self.labels: 128维 list
        for i in range(15):
           self.glabel_dict[i] = i

        self.labels = F.tensor(self.labels)
        # generate one-hot attribute for node according to its label
        nlabel_set = set([])
        for g in self.graphs:
            nlabel_set = nlabel_set.union(
                set([F.as_scalar(nl) for nl in g.ndata['label']])
            )
        nlabel_set = list(nlabel_set)
        label2idx = {
            nlabel_set[i]: i
            for i in range(len(nlabel_set))
        }
        for g in self.graphs:
            attr = np.zeros((
                g.number_of_nodes(),len(nlabel_set)
            ))
            attr[range(g.number_of_nodes()),[label2idx[nl] for nl in F.asnumpy(g.ndata['label']).tolist()]] = 1
            g.ndata['attr'] = F.tensor(attr, F.float32)

        self.gclasses = len(self.glabel_dict)
        self.nclasses = len(self.nlabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata['attr'][0])

        if self.verbose:
            print('Done')
            print(
                """
                -------- Data Statistics --------'
                #Graphs: %d
                #Graph Classes: %d
                #Nodes: %d
                #Node Classes: %d
                #Node Features Dim: %d
                #Edges: %d
                Avg. of #Nodes: %.2f
                Avg. of #Edges: %.2f \n """ % (
                    self.N, self.gclasses, self.n, self.nclasses,
                    self.dim_nfeats, self.m,
                    self.n / self.N, self.m / self.N))

    def save(self):
        graph_path = os.path.join(self.save_dir,f'{self.cityName}_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        label_dict = {'labels': self.labels}
        info_dict = {'N': self.N,
                     'n': self.n,
                     'm': self.m,
                     'gclasses': self.gclasses,
                     'nclasses': self.nclasses,
                     'dim_nfeats': self.dim_nfeats,
                     'glabel_dict': self.glabel_dict,
                     'nlabel_dict': self.nlabel_dict,
                     'graph_ids':self.graphID}

        save_graphs(str(graph_path),self.graphs,label_dict)
        save_info(str(info_path),info_dict)
        pass

    def load(self):
        graph_path = os.path.join(self.save_dir,f'{self.cityName}_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graphs = graphs
        self.labels = label_dict['labels']

        self.N = info_dict['N']
        self.n = info_dict['n']
        self.m = info_dict['m']
        self.gclasses = info_dict['gclasses']
        self.nclasses = info_dict['nclasses']
        self.dim_nfeats = info_dict['dim_nfeats']
        self.glabel_dict = info_dict['glabel_dict']
        self.nlabel_dict = info_dict['nlabel_dict']
        self.graphID = info_dict['graph_ids']

    def has_cache(self):
        graph_path = os.path.join(self.save_dir,f'{self.cityName}_gin.bin')
        info_path = os.path.join(self.save_dir,f'{self.cityName}_gin.pkl')
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    @property
    def num_classes(self):
        return self.gclasses











