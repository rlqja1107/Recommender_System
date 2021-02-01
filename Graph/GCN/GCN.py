import sys
sys.path.append('..')
from dataset import load_data
import torch
<<<<<<< HEAD
=======
import pickle 
>>>>>>> 655920bae508f642bef71ae4829b3088e9c02882
import numpy as np
from math import sqrt
import torch.nn.functional as F
from timeit import default_timer as timer 
from tensorboardX import SummaryWriter
import os

class GCNLayer(torch.nn.Module):
    """
    GCN Hidden Layer
    """
<<<<<<< HEAD
    def __init__(self, input_dim,output_dim,activation=True):
        super(GCNLayer, self).__init__()
=======
    def __init__(self,input_dim,output_dim,activation=True):
        super(GCNLayer,self).__init__()
>>>>>>> 655920bae508f642bef71ae4829b3088e9c02882
        self.linear=torch.nn.Linear(input_dim,output_dim)
        glort_beng=sqrt(6)/sqrt(input_dim+output_dim)
        self.linear.weight.data.uniform_(-glort_beng,glort_beng).float()
        self.activation=torch.nn.ReLU() if activation else None
            
    def forward(self,input_data):
        """
        input : First Layer - Node X 16, Second Layer - 16 X # Label
        """
        output=self.linear(input_data)
        return self.activation(output) if self.activation!=None else output

    
class GCN(torch.nn.Module):
    def __init__(self,config):
        super(GCN,self).__init__()
        self.epoch=config['epoch']
        self.gcn_l1=GCNLayer(config['input_l1_dim'],config['output_l1_dim'])
        self.gcn_l2=GCNLayer(config['output_l1_dim'],config['output_l2_dim'],activation=False)
        self.dropout=torch.nn.Dropout(config['dropout'])
        self.lr=config['lr']
        self.best_accuracy=0.0
        self.max_trial=5
        self.cur_trial=0
        log_dir=os.path.join('cora')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer=SummaryWriter(log_dir)
        
    def preprocess_A_hat(self,adj_mtx):
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
        return torch.from_numpy(A_hat).cuda()
        
    def forward(self, A_hat, feature):
        """
        parmeter : Preprocessed Adjacency matrix, Feature array (Node(V) X # of Feature)
        """
        arr_layer_1=torch.mm(A_hat,feature).float()
        arr_layer_1=self.gcn_l1(arr_layer_1)
        arr_layer_1=self.dropout(arr_layer_1)
        arr_layer_2=torch.mm(A_hat.float(),arr_layer_1)
        arr_layer_2=self.gcn_l2(arr_layer_2)
        return F.log_softmax(arr_layer_2,dim=1)
    
    def accuracy(self,output,label,msk):
        # predict
        predict_class=torch.argmax(output,dim=1)
        # target
        ratio_correct= (predict_class == label)
        correct=torch.sum(ratio_correct.float()*msk)
        return correct/torch.sum(msk)
    
    @staticmethod    
    def run(model,train_msk,valid_msk,test_msk,label,adj,feature_cat):
        label=torch.argmax(label,dim=1)
        A_hat=model.preprocess_A_hat(adj)
        start_total=timer()
        optimizer=torch.optim.Adam(model.parameters(),lr=model.lr)
        for e in range(model.epoch):
            start=timer()
            model.train()
            output=model(A_hat,feature_cat)
            train_loss=F.nll_loss(output[train_msk],label[train_msk])
            model.eval()
            valid_accuracy=model.accuracy(output,label,valid_msk).item()
#             if not model.early_stop(valid_accuracy):
#                 print("Last Epoch : {:d}, accuracy : {:.4f}".format(e+1,valid_accuracy))
#                 break
           
            model.train()
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            model.writer.add_scalars('loss',{'train_loss':train_loss.item()},e)
            model.writer.add_scalars('accuracy',{'valid_loss':valid_accuracy},e)
            if e%20 ==0:
                print("Epoch : {:d}, Train Loss : {:.4f}, Accuracy : {:.4f}, Time : {:.4f}".format(e+1,train_loss.item(),valid_accuracy,timer()-start))
        model.eval()
        output=model(A_hat,feature_cat)
        test_accuracy=F.nll_loss(output[test_msk],label[test_msk]).item()
        print("Accuracy : {:.4f}, Time : {:.4f}".format(test_accuracy,timer()-start_total))
        model.writer.close()
            
            
    def early_stop(self,accur):
        """
        For Early Stop
        """
        if self.best_accuracy<accur:
            self.cur_trial=0
            self.best_accuracy=accur
            return True
        elif self.max_trial>self.cur_trial:
            self.cur_trial+=1
            return True
        else:
            return False
            
    

