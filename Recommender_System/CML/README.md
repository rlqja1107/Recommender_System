# Collaborative Metric Learning   
## Implementation Term  
2021-01-09 ~ 2021-01-17  
## Main Library, Dataset  
**Pytorch**, **[Movielens-1M](https://grouplens.org/datasets/movielens/1m/)**
## Code  
**CML.py** : Class of CML including the related function and static method of running function(**run**)  
**CML_Pytorch.ipynb** : Another implementation of CML not successful learning but, hard try to work one by one.  
**Main_CML.py** : Main function  
**dataset.ipynb** : Implementation of dataset inherited by Dataset.  
## Summary  
### Introduction  
Matrix Factorization is a popular method using dot product of item and user latent vector for calculating the interaction. This approach looks very simple and intuitive, in reality so on. For comparing similar between two object, MF have used the heuristic method that used the **cosine** similarity.  
However, it couldn't be applied to comparing it as **triangle inequality** constraint violated, i.e)**dot product** of two object is equal to zero. In this weakness, Metric learning algorithm using **euclidean** distance approach is more powerful and helpful to compare the similarity of user-item relationship, even item-item and user-user. As metric learning apply to training, the items that a user **co-like** and the users that like the similar item would be **clustered**.  
This is a way user pulls the preferred items including relevant it and push the **imposters** that are inward margin. As &wscr;<sub>ij</sub> is log(rank<sub>ij</sub>+1), rank<sub>ij</sub> is hard to estimate the value. The alternative approach called "Negative Random Sampling" is usually used for saving the time. In this training, negative sampling size is **10**. You can easily change the size from 10 to 20.  
### Model  
##### Objective Function  
 <img width="150" src="https://user-images.githubusercontent.com/55014424/104839970-f1b8df00-5907-11eb-959e-f74d7f631e70.gif">   
  
##### Main Loss funtion   
<img width="250" src="https://user-images.githubusercontent.com/55014424/104839591-79511e80-5905-11eb-8deb-f3f16bba985f.png">  

##### Covariance Regularization    
<img width="180" src="https://user-images.githubusercontent.com/55014424/104840065-7277db00-5908-11eb-975d-0f2b3aea9812.gif">

##### Transformation Regularization  
<img width="180" src="https://user-images.githubusercontent.com/55014424/104842846-c97eaf80-590a-11eb-84ad-b6fdbcf84202.gif">  




## Result  
|Epoch | Total loss | Recall@50 | Recall@100 | Time per epoch(sec) |      
|:---:|:---:|:---:|:---:|:---:|   
|1|5883723.505|0.1171|0.2190|119.80|   
|2|6188269.980|0.1183|0.2309|120.344|     
|3|6373395.147|0.1203|0.2381|120.524|    
|...|...|...|...|... |  
|24 | 5101911.439 | 0.2642 | 0.3927 | 120.731 |    


[Result of Screenshot](https://user-images.githubusercontent.com/55014424/104839031-0f834580-5902-11eb-8fbd-2c9e6c1891d4.png)






