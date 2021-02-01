
from torch.utils.data import DataLoader, Dataset
  
class CML_Dataset(Dataset):  
    def __init__(self,user_id,item_id, neg_item_id):
        self.user_id=user_id
        self.item_id=item_id
        self.neg_item_id=neg_item_id
    
    def __len__(self):
        return len(self.user_id)
    
    def __getitem__(self,index):
        return self.user_id[index], self.item_id[index], self.neg_item_id[index]
