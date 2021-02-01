from node2vec import node2vec


config={'r':10,
        'walk_length':80,
        'dim':128,
        'p':1,
        'q':1,
        'graph_path':'../data/ind.cora', 
        'window':10
}
graph=node2vec(config)
# vector : embedding vector
vector=node2vec.run(graph)
