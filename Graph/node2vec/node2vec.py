import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))
from dataset import load_graph
import random
import numpy as np
from gensim.models import Word2Vec
from timeit import default_timer as timer


class node2vec(object):
    def __init__(self,config):
        self.graph=load_graph(config['graph_path'],weight=False).to_undirected()
        self.r=config['r']
        self.walk_length=config['walk_length'] 
        self.dim=config['dim']
        self.p=config['p']
        self.q=config['q']
        self.window=config['window']
        
        
    def walking(self):
        walks=[]
        nodes=list(self.graph.nodes)
        for i in range(self.r):
            random.shuffle(nodes)
            for n in nodes:
                walks.append(self.node2vec_walk(n))
        return walks
                
    def node2vec_walk(self,src):
        walk=[src]
        while len(walk) < self.walk_length:
            cur_node=walk[-1]
            cur_node_neigh=sorted(self.graph.neighbors(cur_node))
            if len(cur_node_neigh)>0:
                # For walking at first
                if len(walk)==1:
                    walk.append(cur_node_neigh[self.direct_to_node(self.alias_nodes[cur_node][0],self.alias_nodes[cur_node][1])])
                else:
                    pre_node=walk[-2]
                    next_node=cur_node_neigh[self.direct_to_node(self.alias_edges[(pre_node,cur_node)][0],self.alias_edges[(pre_node,cur_node)][1])]
                    walk.append(next_node)
            else:
                break
        return walk    
    
    def preprocess_Modified(self):
        # Weight Setting
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['weight']=1
            self.graph[edge[1]][edge[0]]['weight']=1
            
        alias_nodes={}
        alias_edges={}
        
        # For alias sampling
        # alias nodes would be used when walking starts at first
        for node in self.graph.nodes():
            node_prob=[self.graph[node][n]['weight'] for n in sorted(self.graph.neighbors(node))]
            Z=float(sum(node_prob))
            normalize_prob=[e/Z for e in node_prob]
            alias_nodes[node]=self.alias_setting(normalize_prob)
            
        # For alias sampling 
        # alias edge would be used after the walking at first
        # Undirected Graph
        for edge in self.graph.edges():
            alias_edges[edge]=self.get_alias_edge(edge[0],edge[1])
            alias_edges[(edge[1],edge[0])]=self.get_alias_edge(edge[1],edge[0])
            
        self.alias_nodes=alias_nodes
        self.alias_edges=alias_edges
    
    
    def get_alias_edge(self,src,dst):
        """
        For preprocess the transition probability
        """
        edge_prob=[]
        for dst_neighbor in sorted(self.graph.neighbors(dst)):
            # distance : 0
            if dst_neighbor==src:
                edge_prob.append(self.graph[dst][dst_neighbor]['weight']/self.p)
            # distance : 1
            elif self.graph.has_edge(dst_neighbor,src):
                edge_prob.append(self.graph[dst][dst_neighbor]['weight'])
            # distance : 2
            else:
                edge_prob.append(self.graph[dst][dst_neighbor]['weight']/self.q)
        Z=float(sum(edge_prob))
        norm_prob=[e/Z for e in edge_prob]
        return self.alias_setting(norm_prob)    
    
    def alias_setting(self,prob):
        """
        Alias Method
        reference : https://en.wikipedia.org/wiki/Alias_method (table generation)
        At Last, U_i is filled with value of smaller than 1
        """
        n=len(prob)
        U=np.zeros(n)
        # for floor the value
        K=np.zeros(n, dtype=np.int)
        overfull=[]
        underfull=[]
        for i,p in enumerate(prob):
            U[i]=n*p
            if U[i]<1.0:
                underfull.append(i)
            else:
                overfull.append(i)
                
        while len(overfull)>0 and len(underfull)>0:
            under=underfull.pop()
            over=overfull.pop()
            K[under]=over
            U[over]+=U[under]-1.0
            if U[over]<1.0:
                underfull.append(over)
            else:
                overfull.append(over)
        return K, U
            
        
    def direct_to_node(self,K,U):
        """
        Direct to next node using alias sampling 
        """
        n=len(K)
        node=int(np.floor(np.random.rand()*n))
        return node if np.random.rand()<U[node] else K[node]


    def make_string(self,walks):
        walk=[]
        for i in walks:
            temp=[]
            for j in i:
                temp.append(str(j))
            walk.append(temp)
        return walk
    
    @staticmethod
    def run(graph):
        start=timer()
        print("Preprocess Start")
        graph.preprocess_Modified()
        print("Preprocess Finish, Time :{:.4f}".format(timer()-start))
        start=timer()
        walks=graph.walking()
        print("Walking Finish, Time : {:.4f}".format(timer()-start))
#         flatten = lambda l: [item for sublist in l for item in sublist]
#         walks=[map(str,i) for i in walks]
        walks=graph.make_string(walks)
        start=timer()
        model=Word2Vec(sentences=walks, size=graph.dim,window=graph.window,min_count=0,sg=1,workers=8,iter=1)
        print("SGD Finish, Time : {:.4f}".format(timer()-start))
        return model.wv

    
