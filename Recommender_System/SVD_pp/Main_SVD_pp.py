import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from SVD_pp import SVD_pp
from timeit import default_timer as timer  
import Call_Function as cf
import algo_common_func as ac  

if __name__ =='__main__':
    #default dir : './ml-100k'
    cf.change_dir('../ml-100k')    
        
    start=timer()

    n_user,n_movie,n_rating=cf.read_user_inform()
        
    svd_pp=SVD_pp(n_user,40,n_movie,epoch=20)

    self_instance=svd_pp.get_self_instance()
        
    ac.read_u_data(self_instance,cf.dir_location+'/u.data')

    ac.get_overall_mean(self_instance)
    start=timer()    
    RMSE_list=[]

    for i in range(cf.k_fold):
        svd_pp.gradient_descent(cf.train_set_list[i])
        RMSE=svd_pp.predict(cf.test_set_list[i])
        print(str(i)+" . RMSE :"+str(RMSE))
        RMSE_list.append(RMSE)
    total_time=timer()-start
    print("Time : "+str(total_time))
    print("AVG RMSE : "+str(sum(RMSE_list)/len(RMSE_list)))
    

        
        
        

