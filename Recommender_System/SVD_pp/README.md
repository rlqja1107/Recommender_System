# Singular Value Decomposition ++ (SVD++)  
## Duration  
2020-12-28 ~ 2020-01-04  
## Model  
* r<sub>ui</sub> = &bfr;<sub>ui</sub>+&qscr;<sub>i</sub><sup>T</sup>(&pscr;<sub>u</sub>+|N(u)|<sup>-0.5</sup> &sum;<sub>j∈N(u)</sub>&Yscr;<sub>j</sub> )  

N(u) : set of item that user show the implicit feedback  
&bfr;<sub>ui</sub> : baseline estimate  
&qscr;<sub>i</sub> : the latent factor of item  
&pscr;<sub>u</sub> : the latent factor of user  
&Yscr;<sub>j</sub> : the hidden implicit factor of item(Item X Factor)  
|N(u)|<sup>-0.5</sup> &sum;<sub>j∈N(u)</sub>&Yscr;<sub>j</sub> : implicit feedback of each user  

### Summary    
SVD Model doesn't consider the **implicit feedback** that is usually provided more than explicit feedback. In this Recommendation, when a user rate a item, it is regarded as providing implicit feedback by user. Practically, it is computed by explicit feedback and other behavior is considered as implicit feedback.   
The model could **skew** to users who have rated as possible as many. So, **normalization** have to be put in model. |N(u)|<sup>-0.5</sup> makes latent factor model safe and fair.  
The parameter &pscr;<sub>u</sub>, &qscr;<sub>i</sub>, &Yscr;<sub>j</sub> is learnt by gradient descent methodoloy given train set.   

  
In experiment, epoch is **20**, learning rate(&alpha;) is **0.001** and regulation is **0.001**.   

## Result  
| | RMSE 1 | RMSE 2 | RMSE 3 | RMSE 4| RMSE 5 | AVG RMSE | Total Time  
---|---|---|---|---|---|---|---  
SVD++| 0.98 | 0.96 | 0.955|0.952 | 0.96 | 0.962 | 657sec  

* [Result Of Screenshot](https://user-images.githubusercontent.com/55014424/103785368-742ade80-507e-11eb-8797-71423572b97b.png)   
