import sys 
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from timeit import default_timer as timer
import numpy as np
from math import sqrt
 
class FactorizationMachine(torch.nn.Module):
    def __init__(self,config):
        super(FactorizationMachine,self).__init__()
        self.n_feature=0
        self.weight_decay=config['weight_decay']
        self.n_embedding=config['n_embedding']
        self.lr=config['lr']
        self.batch_size=config['batch_size']
        self.epoch=config['epoch']
        self.is_classifier=config['is_classifier']
        self.loss_func=torch.nn.BCELoss() if self.is_classifier else torch.nn.MSELoss(reduction='sum')
        # for early stop
        self.best_auc=0
        self.best_RMSE=100
        self.back_times=0
        self.limit_trial=3
        
    def forward(self,user_item):
        """
        user_item : batch_size X 2 (user, item)
        """
        # Linear sum
        user_item=user_item+self.offset
        linear_total=torch.sum(self.linear_emb(user_item),dim=1)+self.w_0
        # Quad sum
        x=self.quad_emb(user_item) 
        square_of_sum=torch.sum(x,dim=1)**2
        sum_of_square=torch.sum(x**2,dim=1)
        cal=square_of_sum-sum_of_square
        cal=torch.sum(cal,dim=1,keepdim=True)
        quad_total=0.5*cal
        # Total sum  
        return torch.sigmoid((linear_total+quad_total).squeeze(1)) if self.is_classifier else (linear_total+quad_total).squeeze(1)
    
    
    @staticmethod
    def train_dataset(model,train_loader,optim):
        """
        train_loader : batch X 2 
        optim : Adaptive Gradient Moment
        """
        model.train()
        total_loss=0
        for index,batch in enumerate(train_loader):
            user_item=batch[0]
            target=batch[1]
            #forward 
            y=model(user_item)
            loss=model.loss_func(y,target.float())
            model.zero_grad()
            loss.backward()
            optim.step()
            total_loss+=loss.item()
            
    @staticmethod 
    def test(model,valid_loader):
        """
        valid_loader : batch size X 2(field number)
        """
        model.eval()
        target_list=list()
        predict_list=list()
        with torch.no_grad():
            for index,batch in enumerate(valid_loader):
                user_item=batch[0]
                target=batch[1]
                y=model(user_item)
                target_list.extend(target.tolist())
                predict_list.extend(y.tolist())
        
        return roc_auc_score(target_list,predict_list) if model.is_classifier else sqrt(model.loss_func(torch.tensor(target_list),torch.tensor(predict_list)).item()/len(target_list))
            
        
    def set_embedding(self,dataset):
        feature_dim=dataset.feature_dim.cpu().numpy()
        # w_i
        self.linear_emb=torch.nn.Embedding(torch.tensor(sum(feature_dim)),1).cuda()
        # v_i, v_j
        self.quad_emb=torch.nn.Embedding(torch.tensor(sum(feature_dim)),self.n_embedding).cuda()
        # initialize the weight in embedding
        torch.nn.init.xavier_uniform_(self.quad_emb.weight.data)
        self.w_0=torch.nn.Parameter(torch.rand(1,dtype=torch.float),requires_grad=True).cuda()
        self.offset=torch.tensor(np.array((0,feature_dim[0]))).cuda()
        
    @staticmethod
    def run(model,config,dataset):
        model.set_embedding(dataset)
        train_data, valid_data, test_data = torch.utils.data.random_split(dataset,(dataset.train_length,dataset.valid_length,dataset.test_length))
        train_loader=DataLoader(train_data,batch_size=model.batch_size)
        test_loader=DataLoader(test_data,batch_size=model.batch_size)
        valid_loader=DataLoader(valid_data,batch_size=model.batch_size)
        optim=torch.optim.Adam(model.parameters(),lr=model.lr)
        for e in range(model.epoch):
            start=timer()
            FactorizationMachine.train_dataset(model,train_loader,optim)
            score=FactorizationMachine.test(model,valid_loader)
            if not model.is_continous(score):
                print("Epoch : {:d} Early Stop! Best AUC : {:.4f}".format(e,score))
                break
            print("Epoch : {:d} AUC or RMSE : {:.4f} Time : {:.4f}".format(e+1,score,timer()-start))
        score=FactorizationMachine.test(model,test_loader)
        print("Best AUC or RMSE : {:.4f}".format(score))
    
        
        
    def is_continous(self,cur_score):
        """
        cur_score : Score of ongoing epoch
        For early stop, return False
        is_classifier : False - Regression, True - Classifier
        """
        # Regression
        if not self.is_classifier:
            if cur_score<self.best_RMSE:
                self.best_RMSE=cur_score
                self.back_times=0
                return True
            elif self.limit_trial>self.back_times:
                self.back_times+=1
                return True
            else:
                return False
        # Classifier        
        if cur_score>self.best_auc:
            self.best_auc=cur_score
            self.back_times=0
            return True
        elif self.limit_trial>self.back_times:
            self.back_times+=1
            
            return True
        else:
            return False
        
        
        
