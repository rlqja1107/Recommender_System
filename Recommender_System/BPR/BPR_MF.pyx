
import numpy as np
from timeit import default_timer as timer 
from math import sqrt
import pickle
cimport numpy as np
import algo_common_func as ac
from scipy.sparse import csr_matrix
from math import exp


class BPR_MF:
    def __init__(self,n_user,n_factor,n_movie,epoch,learning_rate=0.001,regulation=0.001,random_state=7):
        self.n_user=n_user
        self.n_factor=n_factor
        self.n_movie=n_movie
        self.rating_matrix=np.zeros((n_user,n_movie),np.int)
        self.name='BPR MF'
        #default value - 100
        self.epochs=epoch
        self.t_rating_matrix=None
        #default value - 1
        self.random_state=random_state
        self.lr=learning_rate
        self.reg=regulation
        self.train_set=None
        self.bi=None
        self.pu=None
        self.qi=None
        self.test_user=0
        
    def get_self_instance(self):
        return self
        
    # init_setting including call function
    def init_setting(self,total_location):
                ac.read_u_data(self,total_location)
    
    def predict_matrix(self):
        cdef np.ndarray[np.double_t,ndim=2] qi=self.qi
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi   
        
        cdef np.ndarray[np.double_t,ndim=2] interaction=np.dot(pu,qi.T)
        
        return interaction
              
    #Get AUC Statistics   
    def AUC(self,test_location):
        cdef np.ndarray[np.int_t,ndim=2] test_matrix=ac.r_test_matrix(self,test_location)
        
        cdef np.ndarray[np.double_t,ndim=2] p_matrix=self.predict_matrix()
        
        cdef np.ndarray[np.double_t,ndim=1] pos_score
        cdef np.ndarray[np.double_t,ndim=1] neg_score
        
        cdef double sum_global=0
        cdef double sum_local=0
        cdef int test_count=0
        for u in range(self.n_user):
                pos_score=p_matrix[u,test_matrix[u]==1]
                if len(pos_score)==0:
                        continue
                neg_score=p_matrix[u,test_matrix[u]==0]
                sum_local=0
                test_count+=1
                for p in pos_score:
                        for n in neg_score:
                                sum_local+=1 if p>n else 0
                sum_global+=sum_local/(len(pos_score)*len(neg_score))
        return sum_global/test_count
        
        
    #item_id : indices of csr_matrix indicating the column of non-zero value   
    def sgd_bootstrap(self,user_id,item_id,neg_item_id,bi,pu,q):
        '''
        Stochastic Gradient Descent Using Bootstrap Sampling
        '''
        cdef long train_len=len(user_id), i_index, j_index
        cdef int i_id, j_id, correct=0, u_id
        cdef int n_factor=self.n_factor
        cdef int n_item=self.n_movie
        cdef double score, temp, lr=self.lr, reg=self.reg, sigmoid
        count=0
        for c in range(train_len):
                i_index, j_index=self.generator(train_len,n_item)
                i_id=item_id[i_index]
                j_id=neg_item_id[j_index]
                u_id=user_id[i_index]
                if self.t_rating_matrix[u_id,item_id[j_index]]!=0:
                        continue
                score=np.dot(pu[u_id],(q[i_id]-q[j_id]))
                sigmoid=1.0/(1.0+exp(score))
                if sigmoid<0.5:
                        correct+=1
                count+=1
                pu_=pu[u_id]
                pu+=lr*(sigmoid*(q[i_id]-q[j_id])-reg*pu_)
                q[i_id]+=lr*(sigmoid*pu_-reg*q[i_id])
                q[j_id]+=lr*(-1*sigmoid*pu_-reg*q[j_id])
                
                bi[i_id]+=lr*(sigmoid-reg*bi[i_id])
                bi[j_id]+=lr*(sigmoid-reg*bi[j_id])
        print("Count "+str(count))
        return pu, q,bi
    

        
    def fit(self,location):
                   
        self.init_vector()
        
        ac.read_train_set(self,location)
        
        csr=csr_matrix(self.t_rating_matrix)    
       
        cdef int epoch=self.epochs
        

        # i_indices : list contatining column of only rated movie id  
        # neg_item_id : all the movie id
        # user_id 
        user_count_t,user_id_t=self.change_csr(csr)
        cdef np.ndarray[np.int_t,ndim=1] neg_item_id=np.arange(self.n_movie,dtype=np.int)
        cdef np.ndarray[np.int_t,ndim=1] user_count=user_count_t
        cdef np.ndarray[np.int_t,ndim=1] user_id=user_id_t
        
        cdef np.ndarray[np.int_t,ndim=1] i_indices=csr.indices.astype(np.int)
        
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=2] q=self.qi
        '''
        Training
        '''
        for i in range(epoch):
                pu, q, bi=self.sgd_bootstrap(user_id,i_indices,neg_item_id,bi,pu,q)
        self.bi=bi
        self.pu=pu
        self.qi=q
                
                
    # generate random number           
    def generator(self,p_len,n_len):
        return np.random.randint(0,p_len),np.random.randint(0,n_len)

    # matrix transform to list of csr
    def change_csr(self,csr):
        
        user_count=np.ediff1d(csr.indptr).astype(np.int)
        user_id=np.repeat(np.arange(self.n_user),user_count).astype(np.int)
        return user_count, user_id  
        
    #initialize all vector by normalizing
    def init_vector(self):
        generator=np.random.RandomState(self.random_state)
        self.bu=np.zeros(self.n_user,np.double)
        self.bi=np.zeros(self.n_movie,np.double)
        self.pu=generator.normal(scale=1.0/self.n_factor,size=(self.n_user,self.n_factor))
        self.qi=generator.normal(scale=1.0/self.n_factor,size=(self.n_movie,self.n_factor))
        
  



