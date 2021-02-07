import os

os.chdir('/home/kibum/recommender_system/Graph/LINE')

import sys

sys.path.append('..')
from dataset import load_graph

import torch


class Line(torch.nn.Module):

    def __init__(self, config):
        super(Line, self).__init__()
        self.graph = load_graph(data_name=config['data_path'], weight=True)
        self.n_dim = config['n_dim']
        self.order = config['order']
        self.node_embed = torch.nn.Embedding(config['n_node'], self.n_dim)
        if self.order == 2:
            self.context_node_embed = torch.nn.Embedding(config['n_node'], self.n_dim)
            self.context_node_embed.weight.data = self.context_node_embed.weight.data.uniform_(-0.5, 0.5) / self.n_dim
        self.node_embed.weight.data = self.node_embed.weight.data.uniform_(-0.5, 0.5) / self.n_dim

    def good(self):
        print("good")
