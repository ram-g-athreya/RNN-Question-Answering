import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from . import Constants
from .tree import Tree


# Dataset class for SICK dataset
class LC_QUAD_Dataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(LC_QUAD_Dataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.pos_sentences = self.read_sentences(os.path.join(path, 'input.pos'))
        self.rels_sentences = self.read_sentences(os.path.join(path, 'input.rels'))
        self.trees = self.read_trees(os.path.join(path, 'input.parents'))

        self.labels = self.read_labels(os.path.join(path, 'output.txt'))
        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        tree = deepcopy(self.trees[index])
        pos_sent = deepcopy(self.pos_sentences[index])
        rels_sent = deepcopy(self.rels_sentences[index])
        label = deepcopy(self.labels[index])
        return (tree, pos_sent, rels_sent, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float, device='cpu')
        return labels
