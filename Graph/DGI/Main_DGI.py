import sys
import os
os.chdir('/home/kibum/recommender_system/Graph/DGI')
sys.path.append(os.path.abspath(os.path.dirname('__file__')))
sys.path.append('..')
# sys.path.append(os.path.abspath(os.path.dirname('__file__'))+'/Graph')
# sys.path.append('..')
from dataset import load_data, normalize, preprocess_adj
from DGI import DeepGraphInformax, LogRegression
import numpy as np
import torch
from timeit import default_timer as timer


def train(epoch, model, optimizer, n_node, feature, loss_function, A_hat, pos_neg_lb, batch_size):
    """
    For training
    Not use mini-batch trainig
    """
    cur_loss = 100000000
    count = 0
    max_count = 20
    start = timer()
    for e in range(epoch):

        model.train()
        optimizer.zero_grad()
        cor_node = np.random.permutation(n_node)
        corrupted_feature = feature[:, cor_node, :]
        logit = model(feature, corrupted_feature, A_hat, n_node, batch_size)
        loss = loss_function(logit, pos_neg_lb)
        if cur_loss > loss:
            cur_loss = loss
            count = 0
        else:
            count += 1
        if count == max_count:
            print("Early Stop, Loss : {:.4f}".format(loss))
            break
        loss.backward()
        optimizer.step()
        if e % 50 == 0:
            print("Epoch : {:d}, Loss : {:.4f}, Time : {:.4f}".format(e, loss, timer()-start))
            start = timer()


def train_one_single_layer(log, optim, cross_entropy, train_pth_rep, train_label, batch_size):
    """
    For train the one single layer parameter
    One Single Layer Output : score each label
    """
    for _ in range(100):
        log.train()
        optimizer.zero_grad()
        logit = log(train_pth_rep.view(batch_size, train_label.shape[0], -1))
        loss = cross_entropy(logit, train_label)
        loss.backward()
        optim.step()


def accuracy(log, test_pth_rep, test_label, batch_size):
    logits = log(test_pth_rep.view(batch_size, test_label.shape[0], -1))
    pred = torch.argmax(logits, dim=1)
    acc = torch.sum(pred == test_label).float() / test_label.shape[0]
    return acc.item()


def test(model, train_msk, test_msk, label, A_hat, feature, hid_dim, batch_size):
    total_index = torch.LongTensor(range(len(train_msk))).cuda()
    train_index = total_index[train_msk]
    test_index = total_index[test_msk]
    valid_index = total_index[valid_msk]

    label_index = torch.argmax(label, axis=1)
    train_label = label_index[train_index]
    test_label = label_index[test_index]
    pth_rep = model.patch_representation(feature, A_hat, total_index.shape[0]).detach()

    train_pth_rep = pth_rep[train_index]
    test_pth_rep = pth_rep[test_index]

    cross_entropy = torch.nn.CrossEntropyLoss()
    accuracy_list = []
    start_test = timer()
    for _ in range(50):
        """
        For averaging the AUC after 50 times 
        """
        log = LogRegression(hid_dim, label.shape[1]).cuda()
        optim = torch.optim.Adam(log.parameters(), lr=0.01)
        train_one_single_layer(log, optim, cross_entropy, train_pth_rep, train_label, batch_size)
        accuracy_list.append(accuracy(log, test_pth_rep, test_label, batch_size)*100)
    print('Average Accuracy: {:.4f}, Test Time: {:.4f}'.format((sum(accuracy_list) / 50), timer()-start_test))


if __name__ == '__main__':
    batch_size = 1
    epoch = 1000
    hid_dim = 512
    adj, feature, _, _, _, train_msk, test_msk, valid_msk, label = load_data(
        data_name='/home/kibum/recommender_system/Graph/data/ind.cora')
    feature = feature.cpu().numpy()
    n_node = feature.shape[0]
    f_size = feature.shape[1]
    adj = preprocess_adj(adj, sparse=True)
    feature = normalize(feature)
    feature = torch.FloatTensor(feature[np.newaxis]).cuda()

    model = DeepGraphInformax(f_size, hid_dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.BCEWithLogitsLoss()

    pos_lb = torch.ones(batch_size, n_node).cuda()
    neg_lb = torch.zeros(batch_size, n_node).cuda()
    pos_neg_lb = torch.cat((pos_lb, neg_lb), 1)
    # Train the model
    train(epoch, model, optimizer, n_node, feature, loss_function, adj, pos_neg_lb, batch_size)
    test(model, train_msk, test_msk, label, adj, feature, hid_dim, batch_size)