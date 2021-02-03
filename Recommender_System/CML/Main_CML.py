

import os
import sys
sys.path.append('/home/kibum/recommender_system/Recommender_System')
from CML import CML
from pathlib import Path
from algo_common_func import split_rating
import torch
from util import set_device_cuda


# Set GPU number 
os.environ["CUDA_VISIBLE_DEVICES"]="3"
rating_path=Path('..')/'ml-1m'/'ratings.dat'
# reading raw  data
train_set, test_set, n_user, n_item=split_rating(rating_path)
config={'n_user':n_item, 
        'n_item':n_user, 
        'n_dim':64, 
        'margin':0.5,
        'epoch':500,
        'batch_size':1024,
        'n_neg_sample':20,
        'lamb_c':10
       }
cml=CML(config).cuda()
cml.init_setting(train_set,test_set)
CML.run(cml)

 
