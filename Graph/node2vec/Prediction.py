from dataset import load_data,load_graph
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

class Prediction(object):
    def __init__(self,feature_vector):
        self.vector=feature_vector
        self.adj, _,_,_,_,_,_,_,label=load_data()
        self.label=label.cpu()
        self.graph=load_graph().to_undirected()
        
    def link_prediction_classifier(self,max_iter=2000):
        lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
        return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])    
    
        
    
    def load_label(path='../data/ind.cora.y'):
        label=None
        with open(path,'rb') as file:
            label=pickle.load(file)
        return label
    
    
    def train_link_prediction_model(self):
        clf=self.link_prediction_classifier()
        link_feature,label=self.link_example_to_feature()
        clf.fit(link_feature,label)
        return clf
        
    def link_example_to_feature(self):
        link_feature=[]
        label=[]
        node=sorted(list(self.graph.nodes()))
        for index,node_1 in enumerate(node):
            for node_2 in node[index,:]:
                if node_1 == node_2:
                    continue
                if self.graph.has_edge(node_1,node_2):
                    label.append(1)
                    link_feature.append(self.binary_operator(node_1,node_2))
                else:
                    label.append(0)
                    link_feature.append(self.binary_operator(node_1,node_2))
        train_link
        return link_feature,label
                
    
    def binary_operator(self,src,dst):
        return np.asarray(self.vector[str(src)])*np.asarray(self.vector[str(dst)])
        