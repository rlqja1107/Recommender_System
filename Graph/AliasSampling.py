import random


def neg_sampling(neg_sample_size, node_sampler, src, dst):
    neg_sample = 0
    neg_list = []
    while neg_sample < neg_sample_size:
        node_sample = node_sampler.sampling(1)[0]
        if node_sample == src or node_sample == dst:
            continue
        else:
            neg_sample += 1
            neg_list.append(node_sample)
    return neg_list


def batch_data(edge_sample, neg_sample_size, node_sampler):
    for src, dst in edge_sample:
        neg_list = neg_sampling(neg_sample_size, node_sampler, src, dst)
        yield [src, dst] + neg_list


class AliasGeneration(object):
    def __init__(self, dist):
        self.dist = dist
        self.prob_table = {}
        self.alias_table = {}
        self.table_index = []
        self.construct_alias_table()

    def construct_alias_table(self):
        """
        reference : https://en.wikipedia.org/wiki/Alias_method
        """
        n = len(self.dist)
        overfull = []
        underfull = []
        scale_prob = {}
        for key, item in self.dist.items():
            scale_prob[key] = item * n
            if scale_prob[key] < 1:
                underfull.append(key)
            else:
                overfull.append(key)

        # Make Alias table
        while len(underfull) > 0 and len(overfull) > 0:
            small = underfull.pop()
            large = overfull.pop()
            self.prob_table[small] = scale_prob[small]
            self.alias_table[small] = large
            scale_prob[large] += scale_prob[small] - 1

            if scale_prob[large] < 1:
                underfull.append(large)
            else:
                overfull.append(large)

        while len(overfull) > 0:
            self.prob_table[overfull.pop()] = 1

        while len(underfull) > 0:
            self.prob_table[underfull.pop()] = 1

        self.table_index = list(self.prob_table.keys())

    def sampling(self, size):
        sample_list = []
        for i in range(size):
            index = random.choice(self.table_index)
            if self.prob_table[index] >= random.uniform(0, 1):
                sample_list.append(index)
            else:
                sample_list.append(self.alias_table[index])

        return sample_list
