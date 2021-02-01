import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
cimport numpy as np 
from timeit import default_timer as timer 
from math import sqrt
from scipy.sparse import csr_matrix
import algo_common_func as ac

class Neighbor_cor:
    #train_set indicate the route of train set
    def __init__(self,n_user,n_movie,epoch,k_nearest=150,learning_rate=0.005,regulation=0.02,random_state=1,lamda=100):
        self.n_user=n_user
        self.n_movie=n_movie
        self.sim_matrix=np.zeros((self.n_movie,self.n_movie),np.double)
        #default value - 100
        self.lamda=lamda
        self.overall_mean=0
        self.epochs=epoch
        #default value - 1
        self.random_state=random_state
        self.lr=learning_rate
        self.reg=regulation
        self.bu=None
        self.bi=None
        self.k=k_nearest
        self.index_matrix=None
        self.t_rating_matrix=None
        self.test_set=None
        
    def get_instance(self):
        return self

            
    def get_mean(self):
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.t_rating_matrix
        self.overall_mean=(rating_matrix.sum())/np.count_nonzero(rating_matrix)
        
    
    def make_sim_matrix(self):
        start=timer()
        cdef int n_movie=self.n_movie
        cdef np.ndarray[np.double_t,ndim=2] sim_matrix=np.zeros((self.n_movie,self.n_movie),np.double)
        cdef np.ndarray[np.int_t,ndim=2] r_m=self.t_rating_matrix
        cdef np.ndarray[np.int_t,ndim=1] n_zero_index
        cdef np.ndarray[np.int_t,ndim=1] dot
        cdef double x_sum =0.0
        cdef double y_sum=0.0, n=0.0
        cdef double xy_sum=0.0
        cdef double coef=0, i_mean, j_mean
        cdef int loading=0
        cdef int lamda =self.lamda
        cdef double result
        for i in range(n_movie):
            for j in range(i,n_movie):
                dot=r_m[:,i]*r_m[:,j]
                n_zero_index=np.where(dot!=0)[0]
                n=len(n_zero_index)
                if i == j or n==0.0:
                        sim_matrix[i,j]=sim_matrix[j,i]=0.0
                        continue
                
                i_mean=r_m[n_zero_index,i].mean()
                j_mean=r_m[n_zero_index,j].mean()
                xy_sum=abs((r_m[n_zero_index,i]*r_m[n_zero_index,j]).sum()-n*i_mean*j_mean)
                if xy_sum==0.0:
                        sim_matrix[i,j]=sim_matrix[j,i]=0.0
                        continue
                x_sum=np.square(r_m[n_zero_index,i]).sum() - n*(i_mean**2)
                y_sum=np.square(r_m[n_zero_index,j]).sum() - n*(j_mean**2)
               
                coef=n/(n+100)
                
                sim_matrix[i,j]=sim_matrix[j,i]=coef*(xy_sum)/sqrt(x_sum*y_sum) if x_sum!=0.0 and y_sum!=0.0 else 0
        self.sim_matrix=sim_matrix
        print("Finish Similarity Time : ",timer()-start)
        
        
    def make_top_sim_matrix(self):
        index_matrix=[]
        cdef np.ndarray[np.int_t,ndim=1] sort_index
        for i in range(self.n_movie):
            sort_index=self.sim_matrix[i].argsort()[::-1]
            index_matrix.append(sort_index)
        self.index_matrix=np.array(index_matrix)
        return index_matrix    
    
    def cost(self,test_location):
        
        ac.read_test_set(self,test_location)
        
        cdef int index, k_, rating, user_index, movie_index, _index
        
        cdef int k=self.k, n_movie=self.n_movie
        
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1]bi=self.bi
        cdef np.ndarray[np.int_t,ndim=2] index_matrix=self.index_matrix
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.t_rating_matrix
        cdef np.ndarray[np.double_t,ndim=2] sim_matrix=self.sim_matrix
        
        cdef double overall_mean=self.overall_mean
        cdef double error, r_ui_hat, neigh_term, sum_bottom, sum_top, baseline
        cdef double total_count=len(self.test_set), err_sum=0.0
        
        for user_index, movie_index, rating in self.test_set:
               
                k_=0
                index=0
                sum_top=0
                sum_bottom=0
                index=0
                while k_<k and index<n_movie:
                        
                        j=index_matrix[movie_index,index]
                        if sim_matrix[movie_index,j]==0.0:
                                break
                        if rating_matrix[user_index, j] != 0:
                                baseline=overall_mean+bu[user_index]+bi[j]
                                sum_top+=sim_matrix[movie_index,j]*(rating_matrix[user_index,j]-baseline)
                                sum_bottom+=sim_matrix[movie_index,j]
                                k_+=1
                        index+=1
                neigh_term=sum_top/sum_bottom if sum_bottom!=0.0 else 0.0
                r_ui_hat=overall_mean+bu[user_index]+bi[movie_index]+neigh_term
                error=rating-r_ui_hat
                err_sum+=(error)**2
        return sqrt(err_sum/total_count)
        
    def fit(self,train_location):
        
        ac.read_train_set(self,train_location)
        
        self.get_mean()
        
        self.make_sim_matrix()

        self.make_top_sim_matrix() 
        
        self.init_vector()
        
        csr=csr_matrix(self.t_rating_matrix)
        
        csr_indptr=csr.indptr
        csr_indices=csr.indices
        csr_rate=csr.data
        
        cdef int k=self.k
        cdef int n_movie=self.n_movie
        cdef np.ndarray[np.double_t,ndim=1] bu=self.bu
        cdef np.ndarray[np.double_t,ndim=1]bi=self.bi
        cdef np.ndarray[np.int_t,ndim=2] index_matrix=self.index_matrix
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.t_rating_matrix
        cdef np.ndarray[np.double_t,ndim=2] sim_matrix=self.sim_matrix
        cdef int epochs=self.epochs
        
        cdef int index, _index, k_
        cdef int i_indptr=0, i_indices=-1, user_index=0, movie_index, rating, no_zero_count
        
        cdef double sum_top, sum_bottom, baseline, r_ui_hat, error, sum_tb
        cdef double lr=self.lr
        cdef double reg=self.reg
        cdef double overall_mean=self.overall_mean
        
        for epoch in range(epochs):
                i_indptr=0
                i_indices=-1
                user_index=0
                while i_indptr+1 < len(csr_indptr):
                        no_zero_count=csr_indptr[i_indptr+1]-csr_indptr[i_indptr]
                        for ent in range(no_zero_count):
                                i_indices+=1
                                movie_index=csr_indices[i_indices]
                                rating=csr_rate[i_indices]
                                k_=0
                                sum_top=0.0
                                sum_bottom=0.0
                                index=0
                                while k_<k and index<n_movie:
                                        _index=index_matrix[movie_index,index]
                                        if rating_matrix[user_index,_index] != 0:
                                                baseline=overall_mean+bu[user_index]+bi[_index]
                                                sum_top+=sim_matrix[movie_index,_index]*(rating_matrix[user_index,_index]-baseline)
                                                sum_bottom+=sim_matrix[movie_index,_index]
                                                k_+=1
                                        index+=1
                                sum_tb=sum_top/sum_bottom if sum_bottom!=0.0 else 0.0
                                r_ui_hat=overall_mean+bu[user_index]+bi[movie_index]+sum_tb  
                                error=rating_matrix[user_index,movie_index]-r_ui_hat
                                bu[user_index]+=lr*(error-reg*bu[user_index])
                                bi[movie_index]+=lr*(error-reg*bi[movie_index]) 
                        user_index+=1
                        i_indptr+=1
                   
        self.bi=bi
        self.bu=bu
        
         #initialize all vector by normalizing
    def init_vector(self):
        self.bu=np.zeros(self.n_user,np.double)
        self.bi=np.zeros(self.n_movie,np.double)
                    
