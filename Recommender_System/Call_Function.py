import pickle

""" 
ml-100k  
"""
# default save - True
save_or_not=True

# Validation Size
k_fold=5


# Total Data Location
dir_location='./ml-100k'
data_location=dir_location+'/u.data'

# Train data location
train_set_list=[]

#Test data location
test_set_list=[]


for i in range(k_fold):
    train_set_list.append(dir_location+'/u'+str(i+1)+'.base')
    test_set_list.append(dir_location+'/u'+str(i+1)+'.test')
        
        
def change_dir(location):
        global dir_location
        dir_location=location
        train_set_list.clear()
        test_set_list.clear()
        for i in range(k_fold):
            train_set_list.append(dir_location+'/u'+str(i+1)+'.base')
            test_set_list.append(dir_location+'/u'+str(i+1)+'.test')
                
def save_object(name,*save_object):
    with open(name,'wb') as file:
        for i in save_object[0]:
            pickle.dump(i,file)
        
# Read Data Information
def read_user_inform():
    user_inform=open(dir_location+'/u.info','r')
    n_user=int(user_inform.readline().split(' ')[0])
    n_movie=int(user_inform.readline().split(' ')[0])
    n_rating=int(user_inform.readline().split(' ')[0])
    user_inform.close()
    return n_user,n_movie,n_rating



                
