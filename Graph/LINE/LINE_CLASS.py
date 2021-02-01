from Graph.dataset import load_graph
import collections
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from Graph.AliasSampling import AliasGeneration, batch_data
from timeit import default_timer as timer


class Line(torch.nn.Module):
    def __init__(self, config):
        """
        :param config:
        order : 1 if context information is not considered, 2 if context information is considered
        n_dim : # dimension
        neg_sample : # negative sampling
        """
        super(Line, self).__init__()
        self.graph = load_graph(data_name=config['data_path'], weight=True)
        self.n_dim = config['n_dim']
        self.order = config['order']
        self.epoch = config['epoch']
        self.batch_size = config['batch_size']
        self.neg_sample_size = config['neg_sample_size']
        self.lr = config['lr']
        self.edge_dist, self.node_dist, self.n_node = self.get_dist()
        self.n_node += 1
        self.edge_sampler = AliasGeneration(self.edge_dist)
        self.node_sampler = AliasGeneration(self.node_dist)

        self.node_embed = torch.nn.Embedding(self.n_node, self.n_dim)
        self.node_embed.weight.data = self.node_embed.weight.data.uniform_(-0.5, 0.5) / self.n_dim
        if self.order == 2:
            self.context_node_embed = torch.nn.Embedding(self.n_node, self.n_dim)
            self.context_node_embed.weight.data = self.context_node_embed.weight.data.uniform_(-0.5, 0.5) / self.n_dim

    def forward(self, batch):
        v_i = self.node_embed(batch[:, 0])
        if self.order == 2:
            v_j = self.context_node_embed(batch[:, 1])
            neg_node_emb = -self.context_node_embed(batch[:, 2:])
        else:
            v_j = self.node_embed(batch[:, 1])
            neg_node_emb = -self.node_embed(batch[:, 2:])
        positive_batch = torch.mul(v_i, v_j)
        observed_edge = F.logsigmoid(torch.sum(positive_batch, dim=1))

        negative_batch = torch.mul(v_i.view(len(v_i), 1, self.n_dim), neg_node_emb)
        neg_edge = torch.sum(F.logsigmoid(torch.sum(negative_batch, dim=2)), dim=1)

        loss = observed_edge + neg_edge
        return -torch.sum(loss)

    def get_dist(self):
        edge_dist = collections.defaultdict(float)
        node_dist = collections.defaultdict(float)

        weight_dist = dict(deepcopy(self.graph.edges()))
        node_degree = collections.defaultdict(int)

        weight_sum = 0
        n_node = 0
        neg_prob_sum = 0
        for edge in self.graph.edges():
            src, dst, weight = edge[0], edge[1], self.graph[edge[0]][edge[1]]['weight']
            node_dist[src] += weight
            edge_dist[(src, dst)] = weight
            node_degree[src] += weight

            neg_prob_sum += np.power(weight, 0.75)
            weight_sum += weight
            node = max(src, dst)
            n_node = node if n_node < node else n_node

        for n, w in node_dist.items():
            node_dist[n] = np.power(w, 0.75) / neg_prob_sum

        for edge, w in edge_dist.items():
            edge_dist[edge] = w / weight_sum

        return edge_dist, node_dist, n_node

    @staticmethod
    def run(model):
        batch_range = int(len(model.edge_dist) / model.batch_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=model.lr)
        for e in range(model.epoch):
            start = timer()
            total_loss = 0
            for b in range(batch_range-1):
                sample_edge = model.edge_sampler.sampling(model.batch_size)
                batch = list(batch_data(sample_edge, model.neg_sample_size, model.node_sampler))

                batch = torch.LongTensor(batch).cuda()
                model.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("Epoch : {:d}, loss : {:.4f}, Time : {:.4f}".format(e+1, total_loss, timer()-start))


