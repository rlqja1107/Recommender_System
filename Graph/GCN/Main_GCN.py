import os
os.chdir('/home/kibum/recommender_system/Graph/GCN')
import sys
sys.path.append('..')
from dataset import load_data
from GCN import GCN


adj, feature_cat,train_label,test_label,valid_label, train_msk, test_msk, valid_msk, label=load_data()
config={
    'epoch':200,
    'input_l1_dim':feature_cat.shape[1],
    'output_l1_dim':16,
    'output_l2_dim':train_label.shape[1],
    'dropout':0.5,
    'lr':0.01
}

gcn=GCN(config).cuda()
GCN.run(gcn,train_msk,valid_msk,test_msk,label,adj,feature_cat)
    
}
gcn=GCN(config).cuda()
GCN.run(gcn,train_msk,valid_msk,test_msk,label,adj,feature_cat)
