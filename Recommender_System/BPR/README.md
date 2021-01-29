# Bayesian Personalized Ranking Of Matrix Factorization(BPR)   
## Implementation Term   
2021-01-04 ~ 2021-01-08  
## Code  
* Main_BPR_MF.py : Main Function  
* BPR_MF.py : Class File including Stochastic Gradient Descent Using Bootstrapping  
## Summary  
### BPR Learning Algorithm  
For BPR Optimization Criterion, I treid to maximize the **posterior**, ln(&rcy;(&theta;|&lt;<sub>u</sub>) = &sum;<sub>(u,i,j)&isin;D<sub>s</sub></sub> ln(&sigma;(&xscr;<sub>uij</sub>) - &lambda;<sub>&theta;</sub>&lowast; &Vert;&theta;&Vert;<sup>2</sup>. This equation is derived by **MLE**   
ln(p(&theta;|&gt;<sub>u</sub>))   
&equals; ln(p(&gt;<sub>u</sub>|&theta;)&lowast;p(&theta;) <- Using Bayesian   


&equals; ln &lpar; &prod;<sub>(u,i,j)&isin;D<sub>s</sub></sub> &sigma;(&xscr;<sub>uij</sub>)&rpar;+ln(p(&theta;))  <- Using MLE   

&equals; &sum;<sub>(u,i,j)&isin;D<sub>s</sub></sub> ln(&sigma;(&xscr;<sub>uij</sub>)) - &lambda;<sub>&theta;</sub>&lowast; &Vert;&theta;&Vert;<sup>2</sup> &lowast;   

 Applying the sigmoid to make a term differentiable( &sigma;(x)= 1/(1+&iecy;<sup>-&xscr;</sup> )), We can use the **stochastic gradient descent** by directing to **maximum** of objective function for finding the better parameter. You can easily get the optimizing model for BPR by derivating above equation. Then, the equation of **Stochastic Gradient Descent** is derived.   
 * &theta; <- &theta; + &alpha;&lowast;&lpar;&iecy;<sup>-&xscr;<sub>uij</sub></sup>  &div; (1 + &iecy;<sup>-&xscr;<sub>uij</sub></sup>) &lowast; &part;/&part;<sub>&theta;</sub>(&xscr;<sub>uij</sub>)+ &lambda;<sub>&theta;</sub>&theta;)    
 
The derivatives of matrix fatorization is (&part;/&part;<sub>&theta;</sub>(&xscr;<sub>uij</sub>) = h<sub>if</sub> - h<sub>jf</sub> if &theta; = &Wscr;<sub>uf</sub>, &Wscr;<sub>uf</sub> if &theta;=h<sub>if</sub>, -&Wscr;<sub>uf</sub> if &theta;=h<sub>jf</sub>)    
 In this &xscr;<sub>uij</sub>, &xscr;<sub>uij</sub> is equal to &xscr;<sub>ui</sub> - &xscr;<sub>uj</sub>. &xscr;<sub>ui</sub> and &xscr;<sub>uj</sub>  indicate the predicted rating by user that shows the implicit feedback and do not show the implicit feedback respectively. The adequate parameter would be drawn out by training the data set as applying the proper regulation(&lambda;<sub>&theta;</sub>) and learning rate parameter(&alpha;).  
 
### How to get AUC indicating criterion  
* AUC(u) = 1 &div; (&VerticalSeparator; &Iopf;<sub>u</sub><sup>+</sup> &VerticalSeparator;&lowast; &VerticalSeparator; &Iopf; \ &VerticalSeparator; &Iopf;<sub>u</sub><sup>+</sup>&VerticalSeparator;) &lowast; &sum;<sub>i&isin; &Iopf;<sub>u</sub><sup>+</sup></sub> &sum;<sub>j&isin; &Iopf; \ &VerticalSeparator; &Iopf;<sub>u</sub><sup>+</sup></sub>  &delta;(&xscr;<sub>uij</sub>&gt;0)  

AUC(u) comes from each user. We need to get average of AUC. Then  
AUC = 1 &div; &VerticalSeparator; U &VerticalSeparator; &lowast; &sum;<sub>u&isin;U</sub> AUC(u)  
* Notion  
  
&Iopf;<sub>u</sub><sup>+</sup> : list of rated item by user,u.  
&Iopf; \ &VerticalSeparator; &Iopf;<sub>u</sub><sup>+</sup> : list of not rated item by user,u.  
&delta;(&xscr;&gt0) : if x>0, then 1, else 0.  


 
## Result  
* Result Of AUC using BPR Matrix Factorization.   

Definition | Value  
---|---  
Iteration | 80000(train data size)  
Total Iteration | 80000 x 20  
Probability Of Choosing (u,i,j)∈D<sub>s</sub>  | 75%   
Time of 80000 Iteration | Average 4.5 sec.   
Time Of getting AUC statistics | 3.3 sec.     
AUC(Area Under ROC Curve) | 0.84  

* [Result Of Screenshot](https://user-images.githubusercontent.com/55014424/103910691-78233300-5148-11eb-9bfd-b8fb1d086ca3.png)
