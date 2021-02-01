import pandas as pd
import sys
sys.path.append('..')
from torch.utils.data import Dataset
from sklearn import preprocessing 
import patsy
from torch.utils.data import Dataset
import torch 
from timeit import default_timer as timer 
class Movie_Set(Dataset):
    
    def __init__(self,movie_path='../ml-1m/movies.dat',rating_path='../ml-1m/ratings.dat',user_path='../ml-1m/users.dat'):
        self.movies=pd.read_csv(movie_path,sep="::",names=["Movie_id","Title","Genres"],engine='python')
        self.users=pd.read_csv(user_path,sep="::",names=["User_id","Gender","Age","Occupation","Zipcode"],engine='python')
        self.ratings=pd.read_csv(rating_path,sep="::",names=['User_id',"Movie_id",'Rating','Timestamp'],engine='python')
        self.train_size=int(len(self.ratings)*0.7)
        self.valid_size=int(len(self.ratings)*0.1)
        self.test_size=len(self.ratings)-self.train_size-self.valid_size
        self.n_user=self.ratings['User_id'].max()+1
        
        self.users_emb_col=[]
        self.users_ohe_col=[]
        self.movies_emb_col=[]
        self.movies_ohe_col=[]
        self.interact_col=[]
        
        self.n_user=self.ratings['User_id'].max()
        self.n_item=self.ratings['Movie_id'].max()
    

    def __len__(self):
        return len(self.ratings)
    
        
    def __getitem__(self,index):
        return self.users_emb[index],self.users_ohe[index],self.movies_emb[index],self.movies_ohe[index]\
    ,self.interact[index],self.y[index]

    
        
    def to_tensor(self):
        """
        users_emb : 1M X [User id, Gender, Age, Occupation]
        users_ohe : 1M X [gender(2), age(7), occupaton(21)] - One Hot Encoding
        movies_emb : 1M X Movie Id
        movies_ohe : 1M X [Movie id] - One Hot Encoding
        interact : 1M X [Gender:Age, Gender:Occupation, Age:Occupation, Genre:Gender, Genre:Age]
        y : 1M X Rating - Target
        """
        self.users_emb=torch.tensor(self.ratings[self.users_emb_col].values).cuda()
        self.users_ohe=torch.tensor(self.ratings[self.users_ohe_col].values,dtype=torch.float).cuda()
        self.movies_emb=torch.tensor(self.ratings[self.movies_emb_col].values).cuda()
        self.movies_ohe=torch.tensor(self.ratings[self.movies_ohe_col].values).cuda()
        self.interact=torch.tensor(self.ratings[self.interact_col].values).cuda()
        self.y=torch.tensor(self.y.values,dtype=torch.float).cuda()
    

    def preprocess(self):
        """
        Preprocess the data by Pandas
        """
        start=timer()
        # Join the rating data with user and movie data.
        self.ratings=self.ratings.merge(self.movies,left_on='Movie_id',right_on='Movie_id')
        self.movies=self.ratings[self.movies.columns]
        self.ratings=self.ratings.merge(self.users,left_on='User_id',right_on='User_id')
        self.users=self.ratings[self.users.columns]
        
        self.y=self.ratings['Rating'].apply(lambda x:1 if x>3 else 0)
        column=['User_id','Gender','Age','Occupation']
        
        # Encode the label starting from 0.
        self.ratings[column]=self.ratings[column].apply(preprocessing.LabelEncoder().fit_transform)
        self.users_emb_col=column
        column=['Movie_id']
        self.ratings[column]=self.ratings[column].apply(preprocessing.LabelEncoder().fit_transform)
        self.movies_emb_col=column
        column=['Gender','Age','Occupation']
        ohe=preprocessing.OneHotEncoder(categories='auto',sparse=False,dtype='uint8')
        
        # Determine the category in column
        ohe.fit(self.ratings[column])
        
        # transform to one-hot encode from fitting column
        data=ohe.transform(self.ratings[column])
        
        self.ratings=pd.concat([self.ratings,pd.DataFrame(data=ohe.transform(self.ratings[column]),columns=ohe.get_feature_names(column))],axis=1)
        self.users_ohe_col=ohe.get_feature_names(column)
        genres=['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        # Put one-hot encoding genre information
        for genre in genres:
            genre=genre.replace('-','')
            column_name='Genre_'+str(genre)
            self.ratings[column_name]=self.ratings['Genres'].apply(lambda x:1 if genre in x else 0)
            self.movies_ohe_col.append(column_name)
            
        # Put Interact term    
        genre_gender_inter=""
        for genre in self.movies_ohe_col:
            genre_gender_inter+='+'+genre+':Gender'
        genre_age_inter=""
        for genre in self.movies_ohe_col:
            genre_age_inter+='+'+genre+':Age'
        interact=patsy.dmatrix("0+ Gender:Age + Gender:Occupation + Age:Occupation"+genre_gender_inter+genre_age_inter,data=self.ratings.astype('object'),return_type='dataframe').astype('uint8')
        self.interact_col=interact.columns
        self.ratings=pd.concat([self.ratings,interact],axis=1)
        
        # Delete useless information
        self.movies.drop(['Title','Genres'],inplace=True,axis=1)
        self.ratings.drop(['Title','Genres','Zipcode'],inplace=True,axis=1)
        print("Preprocess Finish , Time : {:.4f}".format(timer()-start))
        
        
        
        
        
