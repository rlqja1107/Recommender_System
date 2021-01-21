import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from timeit import default_timer as timer 
from math import sqrt
import pickle
import algo_common_func as ac
cimport numpy as np
 

class SVD_pp:
    def __init__(self,n_user,n_factor,n_movie,epoch=20,mean=0,stdev=0.1,learning_rate=0.001,regulation=0.001,random_state=1):
        self.n_user=n_user
        self.n_factor=n_factor
        self.n_movie=n_movie
        self.rating_matrix=np.zeros((n_user,n_movie),np.int)
        self.t_rating_matrix=None
        self.rating_encode=None
        self.overall_mean=0
        self.epochs=epoch
        #default value - 1
        self.random_state=random_state
        self.init_mean=mean
        self.init_stdev=stdev
        self.lr=learning_rate
        self.reg=regulation
        self.bu=None
        self.bi=None
        self.pu=None
        self.yj=None
        self.qi=None
        self.test_set=None
        self.cal_pu=None
        
    # return self object
    def get_self_instance(self):
        return self

    #Testing    
    def predict(self,test_set_location):
        
        ac.read_test_set(self,test_set_location)
        
        self.cal_total_u_factor()
        
        cdef int movie_index, user_index, rating
        
        cdef int n_factor=self.n_factor
        
        cdef double dot, rated_size, r_hat, error
        cdef double overall_mean=self.overall_mean
        cdef double test_count=0
        cdef double r_sum=0
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.t_rating_matrix
        cdef np.ndarray[np.int_t,ndim=2] rating_encode=self.rating_encode
        
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=2] qi=self.qi
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi
        cdef np.ndarray[np.double_t,ndim=2] yj=self.yj
        cdef np.ndarray[np.double_t,ndim=2] cal_pu=self.cal_pu
        cdef np.ndarray[np.int_t,ndim=1] rated_item
        # In line (user_id, item_id, rating)
        for (user_index,movie_index,rating) in self.test_set:
                
                r_hat=overall_mean+bu[user_index]+bi[movie_index]+np.dot(qi[movie_index],cal_pu[user_index])
                error=rating-r_hat
                r_sum+=(error)**2
                test_count+=1
        return sqrt(r_sum/test_count)
    
    def cal_total_u_factor(self):
        #cdef dict total_u_factor=dict()
        cdef np.ndarray[np.int_t,ndim=2] rating_encode=self.rating_encode
        cdef np.ndarray[np.double_t,ndim=2] total_user_factor=np.zeros((self.n_user,self.n_factor),np.double)
        cdef np.ndarray[np.double_t,ndim=1] cal_implicit
        cdef np.ndarray[np.double_t,ndim=2] yj=self.yj
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        for user_id in range(self.n_user) :
                rated_index=np.where(rating_encode[user_id]==1)
                rated_size=sqrt(len(rated_index[0]))
                user_implicit_feedback=yj[rated_index[0],:].sum(axis=0)/rated_size
                
                total_user_factor[user_id]=pu[user_id]+user_implicit_feedback
        self.cal_pu=total_user_factor
        
        
    #Training
    def gradient_descent(self,train_location):
        self.init_vector()
         
        ac.read_train_set(self,train_location) 
        
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.t_rating_matrix
        cdef np.ndarray[np.int_t,ndim=2] rating_encode=self.rating_encode
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=2] qi=self.qi
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi
        cdef np.ndarray[np.double_t,ndim=2] yj=self.yj
        cdef np.ndarray[np.double_t,ndim=1] user_implicit_feedback
        
        cdef int n_factor=self.n_factor
        cdef int movie_id, user_id, rating
        cdef int n_user=self.n_user
        cdef int n_movie=self.n_movie
        
        cdef double overall_mean=self.overall_mean
        cdef double rated_size, r_hat, error
        cdef double reg=self.reg
        cdef double lr=self.lr
        
        for i in range(self.epochs):
                start=timer()
                 # In line (user_id, item_id, rating)
                for user_id in range(n_user):
                        for movie_id in range(n_movie):
                                if rating_encode[user_id,movie_id]>0:
                                        rated_index=np.where(rating_encode[user_id]==1)
                                        rated_size=sqrt(len(rated_index[0]))
                                        user_implicit_feedback=(yj[rated_index[0],:].sum(axis=0))/rated_size
                                        r_hat=overall_mean+bu[user_id]+bi[movie_id]+np.dot(qi[movie_id],(pu[user_id]+user_implicit_feedback))
                        
                                        error=rating_matrix[user_id,movie_id]-r_hat
                        
                                        bu[user_id]+=lr*(error-reg*bu[user_id])
                        
                                        bi[movie_id]+=lr*(error-reg*bi[movie_id])
                    
                                        pu_=pu[user_id].copy()
                                        qi_=qi[movie_id].copy()
                                        pu[user_id]+=lr*(error*qi_-reg*pu_)
                                        qi[movie_id]+=lr*(error*(pu_+user_implicit_feedback/rated_size)-reg*qi_)
                                        yj[rated_index[0],:]+=lr*((error/rated_size)*qi_-reg*yj[rated_index[0],:])
                                        
                #print(str(i+1)+". Epoch Time :"+ str(timer()-start))
                            
        self.bu=bu
        self.bi=bi
        self.qi=qi
        self.pu=pu
        self.yj=yj
        
    #initialize all the vectors by normalizing and making zero                
    def init_vector(self):
        generator=np.random.RandomState(self.random_state)
        self.bu=np.zeros(self.n_user,np.double)
        self.bi=np.zeros(self.n_movie,np.double)
        self.pu=generator.normal(self.init_mean,self.init_stdev,(self.n_user,self.n_factor))
        self.qi=generator.normal(self.init_mean,self.init_stdev,(self.n_movie,self.n_factor))
        self.yj=generator.normal(self.init_mean,self.init_stdev,(self.n_movie,self.n_factor))

