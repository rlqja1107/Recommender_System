import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__')))+'/FM')
from util import cuda_is_avail
from MovieLens1M import MovieLensDataSet
from algo_common_func import read_rating_1m
import os 
from FM import FactorizationMachine
import torch

if cuda_is_avail():
    os.environ["CUDA_VISIBLE_DEVICES"]='3'
   
config={'epoch':100,
       'lr':0.001,
       'batch_size':1024,
       'weight_decay':1e-6,
       'n_embedding':16,
        'is_classifier':True 
       }

rating_data=read_rating_1m()[:,:3]
dataset=MovieLensDataSet(torch.tensor(rating_data).cuda(), config['is_classifier'])
fm=FactorizationMachine(config).cuda()
FactorizationMachine.run(fm,config,dataset)
 
