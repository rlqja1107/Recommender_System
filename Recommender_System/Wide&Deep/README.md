# Wide And Deep Learning For Recommender Systems  
## Implementation Term  
2021-01-16 ~ 2021-01-20  
## Main Framework  
**Pytorch**  
## Model  
* Wide  
<img width="100" src="https://user-images.githubusercontent.com/55014424/105314824-bf59fb00-5c01-11eb-96ba-e7780c537b15.gif">  

w=[w<sub>1</sub>, w<sub>2</sub>, .....,w<sub>d</sub>] - model parameter  
x=[x<sub>1</sub>, x<sub>2</sub>, .... , x<sub>d</sub>] - d feature vector  
x includes the raw feature vector and transformed features that uses the **cross-product** transformation.  
* Deep  
<img width="150" src="https://user-images.githubusercontent.com/55014424/105315498-c0d7f300-5c02-11eb-8f32-89fe09f151e4.gif">  
W : Weights at l-th layer  
b : bias  
f : activation function(In this experiment, use the Relu function)  


* Wide And Deep  

<img width="330" src="https://user-images.githubusercontent.com/55014424/105315919-7014ca00-5c03-11eb-99b5-a7189254eca8.gif">  

Y: binary class   
&sigma; : final activation function of sigmoid  
&straightphi; : cross product transformation of the origin feature  
[x,&straightphi;(x)] : concat of two vector  

## Summary  
The one of the purpose of Recommendation System is to achieve the memorization and generalizaion. In short, **Memorization** have been used for inferring the predictor by considering the historical data. For utilizing the historical data, all of the data sample need to be stored raising the overhead of storage. The cross-product could be efficiently utilized over the sparse feature.  
Generalization could infer the predictor by using correlation between features. It doesn't have to store the input sample comparing to memorization. Generalization is based on **Wide** model while memorization is based on **Deep** model. Wide and Deep model tries to get advantages of merit of memorization and generalization.  
Wide and Deep model is **jointly** trained not ensembled. In this experiment, user's occupation, age, gender cross-product, and movie genre, raw feature is considered to put on a wide model so that the parmeters of wide are trained and user's age, movie id, user id, user gender and user occupation are **embedded** to deep neural network for training the parameters of Deep model. The sum of the result of wide and deep model is transformed by **logistic function**. The parameters of wide and deep model are simultaneously trained by back propagation using **binary cross entropy loss**.  
## Result  
| Epoch | AUC | Time per epoch(sec) |   
|:---:|:---:|:---:|  
|1|0.7741|44.6087|
|2|0.7850|44.8627|  
|3|0.7881|44.8881|  
|4|0.7895|44.7423|  
|...|...|...|  
|18|0.7923|44.0595|
|19|0.7924|46.4061|  
* [Result Of Screenshot](https://user-images.githubusercontent.com/55014424/105141948-3e393000-5b3d-11eb-80e4-350dc48db60b.png)
