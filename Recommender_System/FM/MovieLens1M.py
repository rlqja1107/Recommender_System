import torch.utils.data
import numpy as np
from pathlib import Path
import torch

class MovieLensDataSet(torch.utils.data.Dataset):
        """
        MovieLens 1M Dataset Inherited by torch.utils.data.Dataset
        feature_dim : [user feature dim, movie feature dim] 
        """
        def __init__(self,data,is_classifier):
                self.user_f=torch.tensor(np.array((0,)),dtype=torch.long)
                self.item_f=torch.tensor(np.array((0,)),dtype=torch.long)
                self.train_length=int(len(data)*0.7)
                self.valid_length=int(len(data)*0.1)
                self.test_length=len(data)-self.train_length-self.valid_length
                self.user_item=data[:,:2]-1
                self.feature_dim=torch.max(self.user_item,dim=0)[0]+1
                target=data[:,2]
                if is_classifier:
                        target[target<=3]=0
                        target[target>3]=1
                self.target=target
               
                 
        def __len__(self):
                """
                Should Override this function
                """
                return len(self.target)
        
        
        def __getitem__(self,index):
                """
                Should Override this function
                """
                return self.user_item[index], self.target[index]
        
                
