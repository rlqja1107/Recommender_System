
from Sorec import Sorec
# when run time
import os
os.chdir('/home/kibum/recommender_system/Recommender_System/Sorec')
config={'lamb_c':10,
       'lambda_':0.001,
       "n_user":49290,
       'n_item':139739, 
       'latent_dim':10,
       'test_size':0.01,
       'lr':0.1,
       'batch_size':1024,
       'epoch':100,
       'max_trial':3}
sorec=Sorec(config)
Sorec.run(sorec)
