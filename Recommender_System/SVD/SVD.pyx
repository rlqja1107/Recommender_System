  
import numpy as np
from timeit import default_timer as timer 
from math import sqrt
import pickle
cimport numpy as np
import algo_common_func as ac
from scipy.sparse import csr_matrix

class SVD:
    def __init__(self,n_user,n_factor,n_movie,mean,stdev,epoch,learning_rate=0.005,regulation=0.02,random_state=1):
        self.n_user=n_user
        self.n_factor=n_factor
        self.n_movie=n_movie
        self.rating_matrix=np.zeros((n_user,n_movie),np.int)
        
        #default value - 100
        self.overall_mean=0
        self.epochs=epoch
        
        #default value - 1
        self.random_state=random_state
        self.init_mean=mean
        self.init_stdev=stdev
        self.lr=learning_rate
        self.reg=regulation
        self.train_set=None
        self.test_set=None
        self.bu=None
        self.bi=None
        self.pu=None
        self.qi=None
        
    def get_self_instance(self):
        return self
        
    # init_setting including call function
    def init_setting(self,total_location):
                ac.read_u_data(self,total_location)
                 # get overall mean
                ac.get_overall_mean(self)
                
              
    #Testing    
    def predict(self,test_location):
        ac.read_test_set(self,test_location)
        
        cdef double test_count=0
        cdef double r_sum=0
#         cdef int movie_index
#         cdef int user_index
#         cdef int rating
        cdef double product=0.0
        cdef double error=0.0
        cdef int n_factor=self.n_factor
        cdef np.ndarray[np.double_t,ndim=2] qi=self.qi
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi
        cdef double overall_mean=self.overall_mean
        for rate in self.test_set:
              test_count+=1
              product=np.dot(qi[rate[1]],pu[rate[0]])
              error=rate[2]-(overall_mean+bu[rate[0]]+bi[rate[1]]+product)
              r_sum+=(error)**2
        return sqrt(r_sum/test_count)
    
    #Training
    def gradient_descent(self,location):
                   
        self.init_vector()
        
        ac.read_train_set(self,location)
        
        csr=csr_matrix(self.t_rating_matrix)    
        
        cdef np.ndarray[np.int_t,ndim=1] csr_indptr=csr.indptr.astype(np.int)
        cdef np.ndarray[np.int_t,ndim=1] csr_indices=csr.indices.astype(np.int)
        cdef np.ndarray[np.int_t,ndim=1] csr_data=csr.data.astype(np.int)
        
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=2] qi=self.qi
        cdef np.ndarray[np.double_t,ndim=1] pu_
        cdef np.ndarray[np.double_t,ndim=1] qi_
        
        # no_zero_count : the number of entry that is non-zero on each row
        cdef int movie_index, rating, no_zero_count
        cdef int i_indptr=0
        cdef int i_indices=-1
        cdef int user_index=0
        cdef int n_factor=self.n_factor
        cdef int epoch=self.epochs
        
        cdef double reg=self.reg
        cdef double product
        cdef double overall_mean=self.overall_mean
        cdef double lr=self.lr
        cdef double error=0.0
        
        start=timer()
        for i in range(epoch):
                i_indptr=0
                i_indices=-1
                user_index=0
                while i_indptr+1 < len(csr_indptr):
                        
                        no_zero_count=csr_indptr[i_indptr+1]-csr_indptr[i_indptr]
                        
                        for ent in range(no_zero_count):
                                i_indices+=1
                                movie_index=csr_indices[i_indices]
                                rating=csr_data[i_indices]
                                
                                error=rating-(overall_mean+bu[user_index]+bi[movie_index]+np.dot(qi[movie_index],pu[user_index]))
                                bu[user_index]+=lr*(error-reg*bu[user_index])
                                bi[movie_index]+=lr*(error-reg*bi[movie_index])
                                pu_=pu[user_index]
                                qi_=qi[movie_index]
                                pu[user_index]+=lr*(error*qi_-reg*pu_)
                                qi[movie_index]+=lr*(error*pu_-reg*qi_)
                        i_indptr+=1
                        user_index+=1        
                
        self.bu=bu
        self.bi=bi
        self.pu=pu
        self.qi=qi
                
                
    #initialize all vector by normalizing
    def init_vector(self):
        generator=np.random.RandomState(self.random_state)
        self.bu=np.zeros(self.n_user,np.double)
        self.bi=np.zeros(self.n_movie,np.double)
        self.pu=generator.normal(self.init_mean,self.init_stdev,(self.n_user,self.n_factor))
        self.qi=generator.normal(self.init_mean,self.init_stdev,(self.n_movie,self.n_factor))
        
                    
  



