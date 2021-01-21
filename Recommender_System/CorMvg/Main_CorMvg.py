import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
import timeit
from timeit import default_timer as timer 
from CorMvg import Neighbor_cor
import Call_Function as cf
import pickle
import algo_common_func as ac


if __name__ == '__main__':
        
    cf.change_dir('../ml-100k')    

    n_user,n_movie,n_rating= cf.read_user_inform()
    """
    train_set means route for data set
    n_user : user 수
    n_factor : factor 수
    n_movie : item 수
    epoch : epoch 수
    train_set training 할 data set의 경로
    num_nearest_k=150 : 근접 item 수
    learning_rate=0.005 : learning rate 수
    regulation=0.02 : regulation 
    random_state=1 : seed로 난수로 정하기 위한 변수
    lamda=100 : similarity 구할 때의 람다
    """
  
    neigh=Neighbor_cor(n_user,n_movie,20,k_nearest=150)

    self_instance=neigh.get_instance()
        
    RMSE_list=[]
    start=timer()
    print('CorMvg Start')
    for i in range(cf.k_fold):
        neigh.fit(cf.train_set_list[i])
        RMSE=neigh.cost(cf.test_set_list[i])
        print(str(i+1)+". RMSE : "+str(RMSE))
        RMSE_list.append(RMSE)
    time=timer()-start
    print("Total Train/Fit Time : ",time)
    avg_RMSE=sum(RMSE_list)/len(RMSE_list)
    print("AVG RMSE : ",avg_RMSE)
    

   




