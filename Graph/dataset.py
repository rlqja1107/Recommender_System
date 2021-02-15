import numpy as np
import pickle
import networkx as nx
import torch
import networkx as nx
from pathlib import Path
import scipy.sparse as sp


def load_graph(data_name='../data/ind.cora', weight=False):
        with open(data_name+'.graph', 'rb') as f:
                data = pickle.load(f)
                graph = nx.Graph(data)

        if not weight:
                for edge in graph.edges():
                        graph[edge[0]][edge[1]]['weight'] = 1
        else:
                for edge in graph.edges():
                        graph[edge[0]][edge[1]]['weight'] = np.random.randint(0, 100)
        return graph


def sparse_mx_to_torch_sparse(sparse_mtx):
        """
        sparse matrix to torch sparse matrix
        """
        sparse_mtx = sparse_mtx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mtx.row, sparse_mtx.col)).astype(np.long))
        value = torch.from_numpy(sparse_mtx.data)
        return torch.sparse.FloatTensor(indices, value, torch.Size(sparse_mtx.shape)).cuda()


def preprocess_adj(adj_mtx, sparse=False):
        """
        return - A Hat
        Input - Adjacency Matrix(Node X Node)
        """
        # A_hat
        I=np.eye(adj_mtx.shape[0])
        # Add Self loop
        A_=adj_mtx+I
        D_=np.sum(A_,axis=1)
        D_inv_sqrt=np.power(D_,-0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)]=0.0
        D_inv_sqrt=np.diag(D_inv_sqrt)
        A_hat=np.dot(np.dot(D_inv_sqrt,A_),D_inv_sqrt)
        return sparse_mx_to_torch_sparse(sp.coo_matrix(A_hat)) if sparse else torch.from_numpy(A_hat).cuda()


def load_data(data_name='../data/ind.cora'):
        objects = []
        suffix=['x','y','allx','ally','tx','ty','graph']
        # rename
        for index,s in enumerate(suffix):
                suffix[index] = data_name+'.'+s
        for s in suffix:
                objects.append(pickle.load(open(Path(s),'rb'),encoding='latin1'))
        x, y, allx, ally, tx, ty, graph = objects
        x,allx,tx=x.toarray(),allx.toarray(), tx.toarray()
        test_index=[]
        with open(data_name+'.test.index','rb') as file:
                line = file.readline()
                while len(line) > 0:
                        test_index.append(int(line[0:-1]))
                        line=file.readline()

        max_idx, min_idx=max(test_index),min(test_index)
        # combine the train and test data
        tx_ext=np.zeros((max_idx-min_idx+1,tx.shape[1]))
        feature_cat=np.vstack([allx,tx_ext]).astype(np.float)
        feature_cat[test_index]=tx
        ty_ext=np.zeros((max_idx-min_idx+1,ty.shape[1]))
        label_cat=np.vstack([ally,ty_ext])
        label_cat[test_index]=ty
        adj_matrix=nx.adjacency_matrix(nx.convert.from_dict_of_lists(graph)).toarray()

        # train the labeled data indexing from 0 to 140
        train_idx=range(len(y))
        valid_idx=range(len(y),len(y)+500)
        test_idx=test_index

        # set 1 at train, valid, test index in length of total node
        train_msk=sample_mask(train_idx,label_cat.shape[0])
        valid_msk=sample_mask(valid_idx,label_cat.shape[0])
        test_msk=sample_mask(test_idx,label_cat.shape[0])
        zero=np.zeros(label_cat.shape)
        train_label=zero.copy()
        valid_label=zero.copy()
        test_label=zero.copy()

        #| 0 , 0, .... 0|
        #        .....
        #| 1, 0, 0,  ...|  In train index
        train_label[train_msk,:]=label_cat[train_msk,:]
        valid_label[valid_msk,:]=label_cat[valid_msk,:]
        test_label[test_msk,:]=label_cat[test_msk,:]

        feature_cat=normalize(feature_cat)
        feature_cat=torch.from_numpy(feature_cat).cuda()
        train_msk=torch.from_numpy(train_msk).cuda()
        test_msk=torch.from_numpy(test_msk).cuda()
        valid_msk=torch.from_numpy(valid_msk).cuda()
        label_cat=torch.from_numpy(label_cat).cuda()
        return adj_matrix, feature_cat, train_label, test_label, valid_label, train_msk, test_msk, valid_msk, label_cat



def normalize(feature):
        """
        return : feature array / sum(feature) for each node
        """
        # sum of the number of features
        row_sum=np.sum(feature,axis=1)
        row_sum_diag=np.power(row_sum,-1)
        row_sum_diag[np.isinf(row_sum_diag)]=0.0
        row_sum_inv=np.diag(row_sum_diag)
        return np.dot(row_sum_inv,feature)


def sample_mask(idx,length):
        msk=np.zeros(length)
        msk[idx]=1
        return np.array(msk,dtype=np.bool)
