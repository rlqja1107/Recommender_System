from timeit import default_timer as timer
cimport numpy as np
import numpy as np 
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix,csr_matrix

"""
epinions_dataset 
"""
def r_epinion_trust(path='../epinions_dataset/trust_data.txt'):
        """
        read epinions trust data and substitute the c_ik to trust value incorporating local authority
        return - coo_matrix of trust data
        """
        trust_data=np.loadtxt(path,delimiter=' ',dtype=np.float)
        row=trust_data[:,0]-1
        col=trust_data[:,1]-1
        t=trust_data[:,2]
#        
#         coo_mtx=coo_matrix((t,(row,col)),shape=(n_user,n_user),dtype=np.float)
#         in_degree=np.asarray(coo_mtx.sum(axis=0))
#         out_degree=np.asarray(coo_mtx.sum(axis=1))

        return row, col, t
  
"""
ml-1m ... 
"""
def split_rating(path,test_size=0.2):
        rating_data=np.loadtxt(path,delimiter='::',dtype=np.int)
        n_user=np.max(rating_data[:,0])
        n_item=np.max(rating_data[:,1])
        # For adjusting the id to index
        rating_data[:,0]-=1
        rating_data[:,1]-=1
        rating_data[:,2]=1
       
        train_set,test_set=train_test_split(rating_data,test_size=test_size)
        return train_set, test_set, n_user, n_item


def read_rating_1m(path=Path('..')/'ml-1m'/'ratings.dat'):
        """
        return user_id, item_id, rating, timestamp
        """
        rating_data=np.loadtxt(path,delimiter='::',dtype=np.int)
        return rating_data

        
        
"""
ml-100k
"""

# Reading data for input rating to matrix  
def read_u_data(self,total_location):
        with open(total_location,'r') as f:
                line=f.readline().split('\t')
                while len(line)>1:
                        self.rating_matrix[int(line[0])-1,int(line[1])-1]=int(line[2])
                        line=f.readline().split('\t')

            
# Get Overall Mean                
def get_overall_mean(self):
        start=timer()
        cdef np.ndarray[np.int_t,ndim=2] rating_matrix=self.rating_matrix
        cdef double total_sum=rating_matrix.sum()
        cdef int non_zero_count=np.count_nonzero(rating_matrix)
        self.overall_mean=total_sum/non_zero_count
        print("Time of getting overall mean : ",timer()-start)
        
def read_train_set_csr(self,train_location):
        print("good")

def read_train_set(self,train_location):
        cdef np.ndarray[np.int_t,ndim=2] t_rating_matrix=np.zeros((self.n_user,self.n_movie),np.int)
        cdef np.ndarray[np.int_t,ndim=2] rating_encode_matrix=np.zeros((self.n_user,self.n_movie),np.int)
        cdef int user_id
        cdef int movie_id
        cdef int rating
        with open(train_location,'r') as file:
                line=file.readline().split('\t')
                while len(line)>1:
                        user_id=int(line[0])-1
                        movie_id=int(line[1])-1
                        rating=int(line[2])
                        t_rating_matrix[user_id,movie_id]=rating
                        rating_encode_matrix[user_id,movie_id]=1
                        line=file.readline().split('\t')
                if hasattr(self,'rating_encode'):
                        self.rating_encode=rating_encode_matrix
                self.t_rating_matrix=t_rating_matrix

                
def read_test_set(self,location):
        test_set=[]
        with open(location,'r') as file:
                line=file.readline().split('\t')
                while len(line)>1:
                        test_set.append((int(line[0])-1,int(line[1])-1,int(line[2])))
                        line=file.readline().split('\t')
                self.test_set=test_set 
                
                
def r_test_matrix(self,location):                
        cdef int user_id,movie_id,rating,test_num
        cdef np.ndarray[np.int_t,ndim=2] test_matrix=np.zeros((self.n_user,self.n_movie),np.int)
        
        with open(location,'r') as file:
                
                line=file.readline().split('\t')
                while len(line)>1:
                        user_id=int(line[0])-1
                        movie_id=int(line[1])-1
                        rating=int(line[2])
                        
                        test_matrix[user_id,movie_id]=1
                        line=file.readline().split('\t')
                
        return test_matrix

