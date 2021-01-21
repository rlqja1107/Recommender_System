import os
from Movie_Set import Movie_Set
from Wide_Deep import Wide_And_Deep
os.environ['CUDA_VISIBLE_DEVICES']='2'
dataset=Movie_Set()
dataset.preprocess()
dataset.to_tensor()

config={
    'epoch':100,
    'batch_size':1024,
    'emb_size':32,
    'users_emb':dataset.users_emb,
    'movies_emb':dataset.movies_emb,
    'interact':dataset.interact,
    'movies_ohe':dataset.movies_ohe,
    'n_user':dataset.n_user,
    'lin_size':100,
    'dropout':0.5
}

model=Wide_And_Deep(config).cuda()
print("Training Start")
Wide_And_Deep.run(model,dataset)