import os
import sys 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from timeit import default_timer as timer 
from math import sqrt
import pickle
import Call_Function as cf
from SVD import SVD
import algo_common_func as ac

#default dir : './ml-100k'
cf.change_dir('../ml-100k')

n_user,n_movie,n_rating=cf.read_user_inform()

svd=SVD(n_user,100,n_movie,0,0.1,20)
 
self_instance=svd.get_self_instance()

ac.read_u_data(self_instance,cf.dir_location+'/u.data')

ac.get_overall_mean(self_instance)

RMSE_list=[]
start=timer()
print('SVD Start')
for i in range(cf.k_fold):
        start_train=timer()
        svd.gradient_descent(cf.train_set_list[i])
        #print(i+1,'. fold training time : ',timer()-start_train)
        RMSE=svd.predict(cf.test_set_list[i])
        print(i+1,'. ',RMSE)
        RMSE_list.append(RMSE)
time=timer()-start

print("Time : ",time)

avg_RMSE=sum(RMSE_list)/len(RMSE_list)

print("RMSE : ",avg_RMSE)

if cf.save_or_not:
    cf.save_object("svd_object.p",[svd,RMSE_list,time,avg_RMSE])

    
    
