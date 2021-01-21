# Netflix Recommender System  
## Duration  
2020-12-28 ~   
<<<<<<< HEAD:Recommender_System/README.md
## Implementation  
* [Neighborhood Using Similarity that uses pearson correlation](https://github.com/rlqja1107/Kaist_Recommender_System/tree/master/Netflix_Analysis/CorMvg)  
* [SVD](https://github.com/rlqja1107/Kaist_Recommender_System/tree/master/Netflix_Analysis/SVD)  
* [SVD++](https://github.com/rlqja1107/Kaist_Recommender_System/tree/master/Netflix_Analysis/SVD_pp)  
* Integrated Model(Not Finished)  
* [Bayesian Personalized Ranking - Matrix Factorization](https://github.com/rlqja1107/Kaist_Recommender_System/tree/master/Netflix_Analysis/BPR)  
=======
>>>>>>> b427cdd431374591eaa6d93e439b14e195e66d80:Netflix_Analysis/README.md
## Main Subject  
**Matrix Factorization And Specialized MF For Analysis Of Netflix Data**  
## Data Set  
* [MovieLens](https://grouplens.org/datasets/movielens)  
* In this Analysis, I used 100k data set in ml-100k directory.  
* ml-1m movie-lens data is used for **CML**.  
## Simple Explanation of Code  
* Main_*.py  
These python files are a main file that executes the algorithm.  
* *.pyx  
These cython files are a class file that includes the class containing algorithm method.  
* *.ipynb  
These jupyter notebook files are a jupyter file corresponding to *.pyx file that is different in terms of using python.
## Notice  
1. Integrated Model is not fully implemented. For implementing successfully this model, Code should be corrected. 
2. In some case, the running time for training would be taken longer as using the **batch** gradient descent.  
