from torch.utils.data import DataLoader,random_split
import torch
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score


class Wide_And_Deep(torch.nn.Module):
    def __init__(self,config):
        super(Wide_And_Deep,self).__init__()
        self.loss_func=torch.nn.BCELoss()
        self.epoch=config['epoch']
        self.batch_size=config['batch_size']
        self.emb_size=config['emb_size']
        self.users_emb=config['users_emb']
        self.movies_emb=config['movies_emb']
        self.interact=config['interact']
        self.movie_ohe=config['movies_ohe']
        self.max_trial=3
        self.cur_trial=0
        self.best_score=0
        
        # Deep Neural Component
        self.user_id_emb=torch.nn.Embedding(config['n_user'],self.emb_size)
        self.user_id_emb.weight.data.uniform_(-0.01,0.01)
        self.user_gender_emb=torch.nn.Embedding(len(torch.unique(self.users_emb[:,1])),self.emb_size)
        self.user_gender_emb.weight.data.uniform_(-0.01,0.01)
        self.user_age_emb=torch.nn.Embedding(len(torch.unique(self.users_emb[:,2])),self.emb_size)
        self.user_age_emb.weight.data.uniform_(-0.01,0.01)
        self.user_ocupt_emb=torch.nn.Embedding(len(torch.unique(self.users_emb[:,3])),self.emb_size)
        self.user_ocupt_emb.weight.data.uniform_(-0.01,0.01)
        self.movie_id_emb=torch.nn.Embedding(len(torch.unique(self.movies_emb[:,0])),self.emb_size)
        self.movie_id_emb.weight.data.uniform_(-0.01,0.01)
        
        # Wide
        self.wide_emb=torch.nn.Embedding(len(self.interact[0])+len(self.movie_ohe[0]),1)
        self.bias=torch.nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        
        # Hidden Layer
        self.hi_lay_1=torch.nn.Linear(5*self.emb_size,config['lin_size'])
        self.hi_lay_2=torch.nn.Linear(config['lin_size'],config['lin_size'])
        self.hi_lay_3=torch.nn.Linear(config['lin_size'],config['lin_size'])
        
        # Dropout layer
        self.drop_1=torch.nn.Dropout(p=config['dropout'])
        self.drop_2=torch.nn.Dropout(p=config['dropout'])
        self.drop_3=torch.nn.Dropout(p=config['dropout'])
        
        # Final
        self.last_lay=torch.nn.Linear((self.interact.shape[1])+(self.movie_ohe.shape[1])+(config['lin_size']),1,bias=True)
        
        
    def forward(self,x):
        """
        x : Batch Size(1024) X [(User Id, Gender, Age, Occupation), (User-One Hot), (Movie Id), (Movie-One Hot), (Interact)]
        Interact : Gender:Age, Age:Occupation, Gender:Occupation, Genre:Age, Genre:Gender
        """
        user_emb=x[0]
        user_ohe=x[1]
        movie_emb=x[2]
        movie_ohe=x[3]
        interact=x[4]
        u_id=self.user_id_emb(user_emb[:,0])
        gender=self.user_gender_emb(user_emb[:,1])
        age=self.user_age_emb(user_emb[:,2])
        occupation=self.user_ocupt_emb(user_emb[:,3])
        movie_id=self.movie_id_emb(movie_emb[:,0])
        
        emb=torch.cat([u_id,age,gender,occupation,movie_id],dim=1)
        
        #Deep
        layer_1=torch.nn.functional.relu(self.drop_1(self.hi_lay_1(emb)))
        layer_2=torch.nn.functional.relu(self.drop_2(self.hi_lay_2(layer_1)))
        final_deep=torch.nn.functional.relu(self.drop_3(self.hi_lay_3(layer_2)))
        
        # interact, movie_ohe : Wide 
        # final : Deep + Wide
        final=self.last_lay(torch.cat([interact.float(),movie_ohe.float(),final_deep.float()],dim=1)).squeeze(1)
        return torch.sigmoid(final) 
    
        
    @staticmethod
    def _train(model,train_loader,optim):
        model.train()
        total_loss=0
        for index,batch in enumerate(train_loader):
            y=model(batch)
            loss=model.loss_func(y,batch[5])
            model.zero_grad()
            loss.backward()
            optim.step()
            total_loss+=loss.item()
                
                
    @staticmethod        
    def _test(model,test_loader,optim):
        model.eval()
        target_list=[]
        predict_list=[]
        with torch.no_grad():
            for index,batch in enumerate(test_loader):
                y=model(batch)
                target_list.extend(batch[5].tolist())
                predict_list.extend(y.tolist())
        return roc_auc_score(target_list,predict_list)
                
    @staticmethod
    def run(model,dataset):
        train_size=int(len(dataset)*0.7)
        valid_size=int(len(dataset)*0.1)
        test_size=len(dataset)-train_size-valid_size
        train_data, valid_data, test_data =random_split(dataset,(dataset.train_size,dataset.valid_size,dataset.test_size))
        train_loader=DataLoader(train_data,batch_size=model.batch_size,shuffle=True)
        valid_loader=DataLoader(valid_data,batch_size=model.batch_size,shuffle=True)
        test_loader=DataLoader(test_data,batch_size=model.batch_size,shuffle=True)
        optim=torch.optim.Adagrad(model.parameters(),lr=0.001)
        init_start=timer()
        for e in range(model.epoch):
            start=timer()
            Wide_And_Deep._train(model,train_loader,optim)
            score=Wide_And_Deep._test(model,valid_loader,optim)
            if not model.early_stop(score):
                print("Epoch {:d}, AUC : {:.4f}, Totel Training Time : {:.4f}".format(e,score,timer()-init_start))
                break
            print("Epoch : {:d}, AUC : {:.4f}, Time : {:.4f}".format(e,score,timer()-start))
        print("Trainig Finish")
        score=Wide_And_Deep._test(model,test_loader,optim)
        print("Test AUC :{.4f}".format(score))
    

    def early_stop(self,cur_score):
        """
        Early Stop when current AUC reverse the best AUC  two times
        """
        if self.best_score<=cur_score:
            self.cur_trial=0
            self.best_score=cur_score
            return True
        elif self.cur_trial<self.max_trial:
            self.cur_trial+=1
            return True
        else:
            return False
        
        
    
    
    
    