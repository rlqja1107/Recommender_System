import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
from CML import CML
from pathlib import Path
from algo_common_func import split_rating



# Set GPU number 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
rating_path=Path('..')/'ml-1m'/'ratings.dat'
# reading raw  data
train_set, test_set, n_user, n_item=split_rating(rating_path)
config={'n_user':n_user,
        'n_item':n_item,
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

 
