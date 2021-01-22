# K Nearest Neighborhood Using Similarity Matrix  
## Implementation Term  
2020-12-28 ~ 2021-01-09  
## Code    
* Main_CorMvg.py : Main Executing Function  
* CorMvg.pyx : Class File including to make similarity matrix and gradient descent in "fit" function.  
## Summary  
This K-Nearest-Neighborhood is the approach of Collaborative Filtering. There are two category which are item-oriented and user-oriented approach. Due to scalability and accuracy, 
item-oriented approach have been popular. To use item-oriented approach, similarity matrix are made by using the known ratings provided from same user, u. From now, many researcher tried to make accuarte 
neighborhood models that some of them don't directly  
use similarity matrix but only referred and directly use similarity, s<sub>ij</sub>. 
In this Model, similarity between items are defined by pearson correlation.  
* s<sub>ij</sub> = (&nscr;<sub>ij</sub> / (&nscr;<sub>ij</sub> + &lambda;)) * &rho;<sub>ij</sub> ,   
  
&rho;<sub>ij</sub> : pearson correlation, &nscr;<sub>ij</sub> : the number of user rated both item i and j, &lambda; : usual value 100.  
By Using s<sub>ij</sub>, we can find the top K similar item rated by user. These item set is denoted by S<sup>k</sup>(&iscr;;&Uscr;). The predicted value r^<sub>ui</sub> can be 
written through baseline estimate and similarity between two items rated on user, u.  
* r^<sub>ui</sub> = b<sub>ui</sub> + &sum;<sub>j&isin;S<sup>k</sup>(&iscr;;u)</sub> s<sub>ij</sub>(r<sub>uj</sub> - b<sub>uj</sub>) / (&sum;<sub>j&isin;S<sup>k</sup>(&iscr;;u)</sub> s<sub>ij</sub>  
  
#### **Direction of Development**  
However, This analysis only considered the similar between two i and j items, not total set of neighbor. It could be developed by extending to full set of neighbor.  
Due to the limit of time in internship Program, I couldn't implement neighborhood model that shows the accuarcy and are developed by considering the global weight.    


## Result  
||RMSE 1 | RMSE 2 | RMSE 3 | RMSE 4 | RMSE 5 | AVG RMSE | Total Time | Time per fold  
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
KNN | 0.945 | 0.935 | 0.927 | 0.9274 | 0.929 | 0.932 | 376.87 sec | 59.62    


The most of time are taken by calculating the similarity between two items to make a similarity matrix. The training and calculating of cost times are just taken **4~5** sec. 
* [Result Screenshot](https://user-images.githubusercontent.com/55014424/104084280-13232680-5289-11eb-84f0-2c7df03f442b.png)  
