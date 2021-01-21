import os 
import sys
sys.path.append('..')
from scipy.sparse import csr_matrix
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from dataset import CML_Dataset

class CML(torch.nn.Module):
    def __init__(self,config):
        super(CML,self).__init__()
        self.config=config
        self.n_user=config['n_user']
        self.n_item=config['n_item']
        self.n_dim=config['n_dim']
        self.batch_size=config['batch_size']
        self.epoch=config['epoch']
        self.margin=config['margin']
        self.n_neg_sample=config['n_neg_sample']
        self.lamb_c=config['lamb_c']
        self.test_set=None
        self.train_set=None
        self.neg_item_dic={}
        self.relu=torch.nn.ReLU()
        self.P=torch.nn.Embedding(self.n_user,self.n_dim,max_norm=1)
        self.Q=torch.nn.Embedding(self.n_item,self.n_dim,max_norm=1)
        
    #initialize the data set    
    def init_setting(self,train_set,test_set):
        train_csr=csr_matrix((train_set[:,2],(train_set[:,0],train_set[:,1])))
        total_item=np.arange(self.n_item)
        pos_item_dic={}
        for user,_,_,_ in train_set:
                if user not in pos_item_dic.keys():
                        pos_item_dic[user]=train_csr.getrow(user).nonzero()[1]
                        self.neg_item_dic[user]=np.setdiff1d(total_item,pos_item_dic[user])
        self.test_set=test_set
        self.train_set=torch.tensor(train_set).cuda()
        print("Setting Finish")
        
    # Random Negative Sampling every epoch
    def data_rand_sampling(self):
        neg_item=[]
        for user,_,_,_ in self.train_set:
                neg=np.random.choice(self.neg_item_dic[user.item()],self.n_neg_sample)
                neg_item.append(neg)
        return self.train_set[:,0], self.train_set[:,1], torch.tensor(neg_item).cuda()
        
        
    @staticmethod   
    def run(model):
        optimizer=optim.Adagrad(model.parameters(),lr=0.01)
      
        for epoch in range(model.epoch):
                total_loss=0
                user_id, item_id, neg_item_id=model.data_rand_sampling()
                dataset=CML_Dataset(user_id=user_id, item_id=item_id, neg_item_id=neg_item_id)
                loader=DataLoader(dataset,batch_size=model.batch_size, shuffle=True)
                model.train()
                start=timer()
                for index, batch in enumerate(loader):
                        user=batch[0]
                        item=batch[1]
                        neg_item=batch[2]
                        optimizer.zero_grad()
                        loss=model.loss_m(user,item,neg_item)+model.lamb_c*model.cov_reg(user,item)
                        loss.backward()
                        optimizer.step()
                        total_loss+=loss.item()
                model.eval()
                recall_50, recall_100 = model.recall_at_k(user_id,item_id)
                print("epoch : {:d}, total_loss : {:.3f}, recall_50 : {:.4f}, recall_100 : {:.4f}, time : {:.4f}".format(epoch,total_loss, recall_50, recall_100,timer()-start))
        
    
    # recall at 100, 50
    def recall_at_k(self,user_id,item_id):
        user=self.P(torch.tensor(np.arange(self.n_user)).cuda())
        item=self.Q(torch.tensor(np.arange(self.n_item)).cuda())
        dist=torch.cdist(user,item).cpu().detach()
        
        for u,i,_,_ in self.train_set:
                dist[u.item(),i.item()]=1000.0
                
        top50_id=torch.topk(dist,k=50,dim=1,largest=False)[1].numpy()
        top100_id=torch.topk(dist,k=100,dim=1,largest=False)[1].numpy()
        hit_50=0
        hit_100=0
        count=len(self.test_set)
        for u,it,_,_ in self.test_set:
                hit_50+=1 if it in top50_id[u] else 0
                hit_100+=1 if it in top100_id[u] else 0
                
        return hit_50/count, hit_100/count        
        
        
    # main loss function
    def loss_m(self,user_id,item_id,neg):
        length=len(user_id)
        user_emb=self.P(user_id).view(length,1,self.n_dim)
        item_emb=self.Q(item_id).view(length,1,self.n_dim)
        neg_item_emb=self.Q(neg)
        d_ij=torch.cdist(user_emb,item_emb).view(-1,1)**2
        d_ik=torch.cdist(user_emb,neg_item_emb).view(-1,self.n_neg_sample)**2
        metric=self.relu(self.margin+d_ij-d_ik)
        
        # find imposter
        imposter=[]
        for i in range(length):
            imposter.append(torch.numel(metric[i,metric[i]==0.0]))
        imposter=torch.tensor(imposter).cuda()
        w_ij=torch.log(imposter*self.n_item/self.n_neg_sample+1).view(length,-1)
        loss=torch.sum(w_ij*metric)
        return loss


    # covariance regularization
    def cov_reg(self,user_id,item_id):
        u_latent=self.P(user_id)
        i_latent=self.Q(item_id)
        matrix=torch.cat([u_latent,i_latent],0)
        m_m=matrix.mean(dim=0)
        y=matrix-m_m
        cov=torch.matmul(y.T,y)/(len(user_id)*2)
        loss=(torch.linalg.norm(cov,ord='fro')-torch.linalg.norm(torch.diagonal(cov),ord=2)**2)/self.n_user
        return loss

