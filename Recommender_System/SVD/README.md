# Singular Value Decomposition(SVD)  
## Duration  
2020-12-28 ~ 2021-01-04  
## Model  
* &#114;<sup>^</sup>=&mu; + b<sub>u</sub> + b<sub>i</sub> +&pscr;<sub>u</sub><sup>T</sup>&lowast;&qscr;<sub>i</sub>  
   
&#114;<sup>^</sup> : predicted rating  
b<sub>u</sub> : baseline estimate of user  
b<sub>i</sub> : baseline estimate of item   
&pscr;<sub>u</sub><sup>T</sup> : user latent factor  
&qscr;<sub>i</sub> : item latent factor  
## Summary  
Latent factor model is alternative of **Collaborative Filtering** approach. This model shows more accuracy and scalability than neighborhood model. Then, Many people have preferred it to use thie model.  
Latent factor model only uses the observed rating comparing to previous SVD. It would be likely to overfitting to the train set. This is reason why regularization model are considered.  
* min<sub>&pscr;, &qscr;, &bfr;</sub> &sum;<sub>(u,i)&isin;K</sub> (&rfr;<sub>ui</sub> - &mu; - &bfr;<sub>u</sub> - &bfr;<sub>i</sub> - &pscr;<sub>u</sub><sup>T</sup>&qscr;<sub>i</sub>)<sup>2</sup> + &lambda;( &Vert; &pscr;<sub>u</sub> &Vert;<sup>2</sup> + &Vert;&qscr;<sub>i</sub>&Vert; + &bfr;<sub>u</sub><sup>2</sup> + &bfr;<sub>i</sub><sup>2</sup>)  

To successfully solve this problem, Full gradient descent method is suggested to direct the minimum of objective function step by step. &lambda; helps to not overfit to train set by setting lower value. This term is called by regularization term. 
### Vulnerable point  
Above latent factor model only includes explicit feedback, not implicit feedback. In reality, Implicit feedback like browsing or click site could easily be considered and taken than explicit feedback. Therefore, Implicit feedback term should be described in latent factor model.  
## Result  
* Result Of SVD Using Cython  
 
Cython/Python | RMSE 1 | RMSE 2 | RMSE 3 | RMSE 4 | RMSE 5 | Avg RMSE | Total Time |   
---|---|---|---|---|---|---|--- |   
Cython | 0.954 |0.941 | 0.938 | 0.935 | 0.933 | 0.941 | <span style="color:red">289.95sec</span> |   
Python | 0.961 | 0.949 | 0.942 | 0.939 | 0.943 | 0.949 | <span style="color:red">6042.4sec</span> |     


Above table shows the necessity and importance of using Cython rather than python.  
Running time using Cython is faster **20** times  
* [Result Of Image](https://user-images.githubusercontent.com/55014424/103980646-99753500-51c3-11eb-821b-73b5cc2ca517.png)  
