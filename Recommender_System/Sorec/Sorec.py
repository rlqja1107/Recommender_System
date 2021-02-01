import sys
sys.path.append('..')
import numpy as np
from math import sqrt,exp, fabs
import torch
from torch.utils.data import DataLoader 
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix,csr_matrix

class Sorec(object):
    
    def __init__(self,config):
        """
        lambda c : 10
        other lambda : 0.001
        """
        self.lamb_c=config['lamb_c']
        self.lambda_=config['lambda_']
        self.n_user=config['n_user']
        self.n_item=config['n_item']
        self.l_dim=config['latent_dim']
        self.test_size=config['test_size']
        self.U=np.random.normal(scale=1.0/self.l_dim,size=(self.l_dim,self.n_user))
        self.V=np.random.normal(scale=1.0/self.l_dim,size=(self.l_dim,self.n_item))
        self.Z=np.random.normal(scale=1.0/self.l_dim,size=(self.l_dim,self.n_user))
        self.lr=config['lr']
        self.batch_size=config['batch_size']
        self.epoch=config['epoch']
        self.best_MAE=100
        self.max_trial=config['max_trial']
        self.cur_trial=0
        
        
    def get_rating_matrix(self):
        """
        return csr_matrix
        """
        rating_data=np.loadtxt('../epinions_dataset/ratings_data.txt',delimiter=' ',dtype=np.float)
        user=rating_data[:,0]-1
        item=rating_data[:,1]-1
        rating=rating_data[:,2]
        rating_csr=csr_matrix((rating,(user,item)),shape=(self.n_user,self.n_item),dtype=np.float)
        train,test=train_test_split(rating_csr,test_size=self.test_size)
        return train, test
        
    def get_trust_matrix(self):
        """
        return coo_matrix
        """
        trust_data=np.loadtxt('../epinions_dataset/trust_data.txt',delimiter=' ',dtype=np.float)
        row=trust_data[:,0]-1
        col=trust_data[:,1]-1
        t=trust_data[:,2]
        trust_coo=coo_matrix((t,(row,col)),shape=(self.n_user,self.n_user))
        in_degree=trust_coo.sum(axis=0)
        out_degree=trust_coo.sum(axis=1)
        for i in range(trust_coo.data.shape[0]):
            row_=trust_coo.row[i]
            col_=trust_coo.col[i]
            trust_coo.data[i]=sqrt(in_degree[0,col_]/(out_degree[row_,0]+in_degree[0,col_]))
        return trust_coo
    
    @staticmethod
    def run(model):
        trust_mat=model.get_trust_matrix()
        train_set, test_set=model.get_rating_matrix()
        print("Datasetting Finish, Train : {:d}, Test : {:d}".format(train_set.data.shape[0],test_set.data.shape[0]))
        for e in range(model.epoch):
            start=timer()
            model.train(train_set,trust_mat)
            mae=model.test(test_set)
            if not model.early_stop(mae):
                print("Final Epoch : {:d}, MAE : {:.4f}".format(e+1,mae))
                break
            print("Epoch : {:d}, MAE : {:.4f}, Time : {:.4f}".format(e+1,mae,timer()-start))
            
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def deri_sig(self,x):
        val=self.sigmoid(x)
        return val*(1-val)
    
    def train(self,train_set,trust_mat):
        r_index=train_set.nonzero()
        c_index=trust_mat.nonzero()
        r_data=(train_set.data-1)/4
        c_data=trust_mat.data
        UV=np.empty(r_data.shape[0])
        UZ=np.empty(c_data.shape[0])
        # csr에서 non-missing value에 대해서만 길게 계산
        for k in range(r_data.shape[0]):
            UV[k]=np.dot(self.U[:,r_index[0][k]].T,self.V[:,r_index[1][k]])
                
        for k in range(c_data.shape[0]):
            UZ[k]=np.dot(self.U[:,c_index[0][k]].T,self.Z[:,c_index[1][k]])
            
        UV=csr_matrix((self.deri_sig(UV)*(self.sigmoid(UV)-r_data),r_index),shape=(self.n_user,self.n_item))
        UZ=csr_matrix((self.deri_sig(UZ)*(self.sigmoid(UZ)-c_data),c_index),shape=(self.n_user,self.n_user))
        
        U=csr_matrix(self.U)
        V=csr_matrix(self.V)
        Z=csr_matrix(self.Z)
        
        grad_U=UV.dot(V.T).T+self.lamb_c*UZ.dot(Z.T).T+self.lambda_*U
        grad_V=UV.T.dot(U.T).T+self.lambda_*V
        grad_Z=self.lamb_c*UZ.T.dot(U.T).T+self.lambda_*Z
        
        self.U=self.U-self.lr*grad_U
        self.V=self.V-self.lr*grad_V
        self.Z=self.Z-self.lr*grad_Z
        
    def test(self,test_set):
        t_index=test_set.nonzero()
        data=test_set.data
        total=data.shape[0]
        loss_numerator=0.0
        for k in range(total):
            pred=4*self.sigmoid(np.dot(self.U[:,t_index[0][k]].T,self.V[:,t_index[1][k]]))+1
            loss_numerator+=fabs(pred-data[k])
        return loss_numerator/total
    
    
    def early_stop(self,mae):
        if mae<self.best_MAE:
            self.cur_trial=0
            self.best_MAE=mae
            return True
        elif self.cur_trial<self.max_trial:
            self.cur_trial+=1
            return True
        else:
            return False
