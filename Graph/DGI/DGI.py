import sys
import os
sys.path.append(os.path.abspath(os.path.dirname('__file__')))
from GCN import GCN
import torch


class LogRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        # self.W.bias.data.fill_(0.0)

    def forward(self, feature):
        result = self.linear(feature)
        return torch.squeeze(result)


class DeepGraphInformax(torch.nn.Module):
    def __init__(self, n_input, n_hidden):
        super(DeepGraphInformax, self).__init__()
        self.gcn = GCN(n_input, n_hidden)
        self.sigmoid = torch.nn.Sigmoid()
        self.PReLu = torch.nn.PReLU()
        self.discriminator = torch.nn.Bilinear(n_hidden, n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.discriminator.weight.data)

    def forward(self, positive_feature, corrupt_feature, A_hat, n_node, batch_size):
        """
        positive_feature : batch X Node X Feature
        corrupt_feature : batch X Node X Feature
        """
        positive_h = self.gcn(positive_feature, A_hat, n_node).view(batch_size, n_node, -1)
        summary_vector = self.sigmoid(torch.mean(positive_h, dim=1)).view(batch_size, 1, -1)
        corrupt_h = self.gcn(corrupt_feature, A_hat, n_node).view(batch_size, n_node, -1)
        summary_vector = summary_vector.expand_as(positive_h)
        D_pos = self.PReLu(self.discriminator(positive_h, summary_vector))
        D_pos = torch.squeeze(D_pos, 2)
        D_corr = self.PReLu(self.discriminator(corrupt_h, summary_vector))
        D_corr = torch.squeeze(D_corr, 2)
        return torch.cat((D_pos, D_corr), 1)

    def patch_representation(self, feature, A_hat, n_node):
        h_i = self.gcn(feature, A_hat, n_node)
        return h_i