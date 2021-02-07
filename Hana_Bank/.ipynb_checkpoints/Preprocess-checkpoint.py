import pandas as pd
import os
import sys
os.path.join('/home/kibum/recommender_system/Hana_Bank')
sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Hana_Bank')
data = pd.read_excel(os.getcwd()+'/Hana_Bank/Hana_Data.xlsx')
data.head()