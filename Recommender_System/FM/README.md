# Factorization Machine  
## Term  
2021-01-18 ~ 2021-01-19  
## Model  
* Original Model  
<img width="300" src="https://user-images.githubusercontent.com/55014424/105289978-9c2c4d00-5bfb-11eb-90e8-2b5133443146.gif">   


* Revised Model For Linear Time Complexity  
<img width="350" src="https://user-images.githubusercontent.com/55014424/105291996-1b218580-5bfc-11eb-9b0f-f411f707c83f.gif">  

w<sub>0</sub> : global bias  
w<sub>i</sub> : weight of each feature space  
v<sub>i</sub>, v<sub>j</sub> : Alternative expression of w<sub>i</sub> using latent feature  
n : total (train) data  
k : dimension of feature space  
## Summary  
Factorization Machine is used for **general predictor** no matter what feature vecotor is and **huge sparsity** as w<sub>i</sub> substitue to v<sub>i</sub> vector for describing the interaction between object. Another merit of it is a model that linearly increase the time complexity by model parameter, so it could be applicable for large dataset.  
Above model of 2 way FM captures the single and pairwise interaction. In pairwise interaction, the independence between weight of each feature is broken by factorizing the weights. For example, v<sub>1</sub> and v<sub>2</sub> interaction would affect to interaction of v<sub>1</sub> and v<sub>3</sub> although v<sub>2</sub> and v<sub>3</sub> is not directly interacted. Following by this step, Sparse rating datas affect not only each interaction but also other interaction. This factorized interaction makes this model performance better and increases the accuracy.  
As you wish to use indicator of predictor, you can use this model as regression, binary classification, and ranking as you want. FM mimic the other state-of-art approach by controlling the parameter, SVD++, Matrix Factorization, and Factorized Personalized Markov Chains.  
#### For your Key points!!  
* Using for **Huge Sparsity**  
* It doesn't matter on what type of input data is as this model is a **General Predictor**!  
* Be applicable for large Dataset for **linearly** increasing time complexity.  

## Result  
* RMSE - Regression  


| Epoch | RMSE | timer per epoch(sec) |
|:---:|:---:|:---:|  
| 1 | 1.723 | 7.8 |  
| ... | ... | ... |  
| 38 | 0.8768 | 7.9401 |
| 39 | 0.8781 | 7.945 |  
| 40 | 0.8796 | 7.8999 |  
| 41 | 0.8811 | 7.9665 |   

[Result Of Screenshot](https://user-images.githubusercontent.com/55014424/104977421-d7375080-5a42-11eb-931c-3e212c4154e5.png)  


* AUC - Classifier   


| Epoch | AUC | timer per epoch(sec) |  
|:---:|:---:|:---:|  
| 1 | 0.6174 | 10.93 |  
| ... | ... | ... |  
| 20 | 0.8069 | 10.9085 |  
| 21 | 0.8066 | 9.6512 |   
| 22 | 0.8062 | 8.6972 |   
| 23 | 0.8056 | 8.7765 |   
 
 [Result Of Screenshot](https://user-images.githubusercontent.com/55014424/104977649-5af13d00-5a43-11eb-98b8-955e3e9edb4d.png)
