
import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
import numpy as np
from timeit import default_timer as timer 
import Call_Function as cf
from SVD import SVD
import algo_common_func as ac

from BPR_MF import BPR_MF


#default dir : './ml-100k'
cf.change_dir('../ml-100k')

n_user,n_movie,n_rating=cf.read_user_inform()

bpr=BPR_MF(n_user,40,n_movie,20)

self_instance=bpr.get_self_instance()

start=timer()
print('BPR MF Start')
bpr.fit(cf.dir_location+'/u1.base')
print("Fit time : "+str(timer()-start))
start=timer()
auc=bpr.AUC(cf.dir_location+'/u1.test')
print('AUC: '+str(auc))    
     
time=timer()-start

print("Time : ",time)



# if cf.save_or_not:
#     cf.save_object("svd_object.p",[svd,RMSE_list,time,avg_RMSE])

