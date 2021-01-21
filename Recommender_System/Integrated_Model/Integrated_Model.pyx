import numpy as np
from timeit import default_timer as timer 
from math import sqrt
import pickle
cimport numpy as np

# In[5]:


class Integrated_Model:
    def __init__(self,n_user,n_factor,n_movie,mean=0,stdev=0.1,lr_b=0.007,lr_other=0.001,epoch=20,reg_b=0.005,reg_other=0.015,random_state=1):
        self.n_user=n_user
        self.n_factor=n_factor
        self.n_movie=n_movie
        self.rating_matrix=np.zeros((n_user,n_movie),np.int)
        self.encode_rating_matrix=np.zeros((n_user,n_movie),np.int)
        
        self.overall_mean=0
        self.epochs=epoch
        
        self.init_mean=mean
        self.init_stdev=stdev
        self.lr_b=lr_b
        self.lr_other=lr_other
        self.reg_b=reg_b
        self.reg_other=reg_other
        #default value - 1
        self.random_state=random_state
        self.bu=None
        self.bi=None
        self.pu=None
        self.qi=None
        self.w=None
        self.c=None
        self.yj=None
        
    def read_u_data(self,total_location):
        with open(total_location,'r') as f:
            print("Read Start")
            line=f.readline().split('\t')
            while len(line)>1:
                self.rating_matrix[int(line[0])-1,int(line[1])-1]=int(line[2])
                self.encode_rating_matrix[int(line[0])-1,int(line[1])-1]=1
                line=f.readline().split('\t')
            print("Finish")
            
    def get_overall_mean(self):
        start=timer()
        total_count=0
        total_sum=0
        for j in range(self.n_movie):
            for i in range(self.n_user):
                if self.rating_matrix[i,j] != 0:
                    total_count+=1
                    total_sum+=self.rating_matrix[i,j]
        self.overall_mean=total_sum/total_count
        print("Time : ",timer()-start)
        
        
    def predict(self,test_set):
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=2] qi=self.qi
        cdef np.ndarray[np.double_t,ndim=2] c=self.c
        cdef np.ndarray[np.double_t,ndim=2] w=self.w
        cdef np.ndarray[np.double_t,ndim=2] yj=self.yj
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.rating_matrix
        cdef np.ndarray[np.int_t,ndim=1] rated_item
        cdef int movie_index
        cdef int user_index
        cdef int rating
        cdef double rated_size
        cdef int n_factor=self.n_factor
        cdef double test_count=0
        cdef double r_sum=0.0
        cdef double overall_mean=self.overall_mean
        with open(test_set,'r') as file:
            line=file.readline().split('\t')
            
            while len(line) >1:
                movie_index=int(line[1])-1
                user_index=int(line[0])-1
                rating=int(line[2])
                user_implicit_feedback=np.zeros(n_factor,np.double)
                rated_item=[item_id for item_id,r in enumerate(rating_matrix[user_index]) if r!=0]
                rated_size=sqrt(len(rated_item))
                for item_id in rated_item:
                    for f in range(n_factor):
                        user_implicit_feedback[f]+=yj[item_id,f]/rated_size
                dot=0
                for f in range(n_factor):
                    dot+=qi[movie_index,f]*(pu[user_index,f]+user_implicit_feedback[f])
                r_hat=overall_mean+bu[user_index]+bi[movie_index]+dot
                error=rating-r_hat
                r_sum+=(error)**2
                test_count+=1
                line=file.readline().split('\t')
        return sqrt(r_sum/test_count)
    
    
    def gradient_descent(self,train_set):
        Integrated_Model.init_vector(self)
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1] bi=self.bi
        cdef np.ndarray[np.double_t,ndim=2] pu=self.pu
        cdef np.ndarray[np.double_t,ndim=2] qi=self.qi
        cdef np.ndarray[np.double_t,ndim=2] c=self.c
        cdef np.ndarray[np.double_t,ndim=2] w=self.w
        cdef np.ndarray[np.double_t,ndim=2] yj=self.yj
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.rating_matrix
        cdef np.ndarray[np.int_t,ndim=1] rated_item
        cdef int movie_index
        cdef int user_index
        cdef int rating
        cdef double rated_size
        cdef int n_factor=self.n_factor
        
        for i in range(self.epochs):
            with open(train_set,'r') as file:
                line=file.readline().split('\t')
                while len(line) >1:
                    movie_index=int(line[1])-1
                    user_index=int(line[0])-1
                    rating=int(line[2])
                    user_implicit_feedback=np.zeros(n_factor,np.double)
                    rated_item=np.array([item_id for item_id,r in enumerate(rating_matrix[user_index]) if r!=0])
                    rated_size=sqrt(len(rated_item))
                    for item_id in rated_item:
                        for f in range(n_factor):
                            user_implicit_feedback[f]+=yj[item_id,f]/rated_size
                    dot=0
                    for f in range(self.n_factor):
                        dot+=self.qi[movie_index,f]*(self.pu[user_index,f]+user_implicit_feedback[f])
                    r_hat=self.overall_mean+self.bu[user_index]+self.bi[movie_index]+dot
                    error=rating-r_hat
                    self.bu[user_index]+=self.lr*(error-self.reg*self.bu[user_index])
                    self.bi[movie_index]+=self.lr*(error-self.reg*self.bi[movie_index])
                    for i in range(self.n_factor):
                        pu_=self.pu[user_index,i]
                        qi_=self.qi[movie_index,i]
                        self.pu[user_index,i]+=self.lr*(error*qi_-self.reg*pu_)
                        self.qi[movie_index,i]+=self.lr*(error*(pu_+user_implicit_feedback[i])-self.reg*qi_)
                        for j in rated_item:
                            self.yj[j,i]+=self.lr*((error/rated_size)*qi_-self.reg*self.yj[j,i])
                    line=file.readline().split('\t')
        
                    
                    
    def init_vector(self):
        generator=np.random.RandomState(self.random_state)
        self.bu=np.zeros(self.n_user,np.double)
        self.bi=np.zeros(self.n_movie,np.double)
        self.pu=generator.normal(self.init_mean,self.init_stdev,(self.n_user,self.n_factor))
        self.qi=generator.normal(self.init_mean,self.init_stdev,(self.n_movie,self.n_factor))
        self.c=generator.normal(self.init_mean,self.init_stdev,(self.n_movie,self.n_movie))
        self.w=generator.normal(self.init_mean,self.init_stdev,(self.n.movie,self.n_movie))
        self.yj=generator.normal(self.init_mean,self.init_stdev,(self.n.movie,self.n_factor))
        

