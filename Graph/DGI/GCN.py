import torch


class GCN(torch.nn.Module):
    def __init__(self, n_input, n_hidden):
        """
        :param n_input: 1433
        :param n_hidden: 512
        """
        super(GCN, self).__init__()
        self.W = torch.nn.Linear(n_input, n_hidden, bias=False)
        self.activation = torch.nn.PReLU()
        torch.nn.init.xavier_uniform_(self.W.weight.data)

    def forward(self, input_arr, A_hat, n_node):
        """
        :param input_arr: 1 X Node X Features
        :return patch representation, # Node X # Hidden layer
        """
        X_THETA = self.W(input_arr)
        X_THETA = torch.squeeze(X_THETA)
        gcn_result = torch.mm(A_hat, X_THETA)
        return self.activation(gcn_result)