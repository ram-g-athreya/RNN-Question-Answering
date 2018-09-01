import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from . import utils
from . import Constants

# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, num_classes, criterion, vocab_output):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.num_classes = num_classes

        self.ix = nn.Linear(self.in_dim, self.mem_dim)
        self.ih = nn.Linear(self.mem_dim, self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)

        self.ux = nn.Linear(self.in_dim, self.mem_dim)
        self.uh = nn.Linear(self.mem_dim, self.mem_dim)

        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        self.oh = nn.Linear(self.mem_dim, self.mem_dim)

        self.ah = nn.Linear(self.mem_dim, self.num_classes)

        self.criterion = criterion
        self.output_module = None
        self.vocab_output = vocab_output

    def set_output_module(self, output_module):
        self.output_module = output_module

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)

        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_sum))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)

        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs, training = False):
        outputs = []
        for idx in range(tree.num_children):
            outputs.append(self.forward(tree.children[idx], inputs, training))

        child_c, child_h = self.get_child_states(tree)
        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h) # TREE.IDX veruses TREE.IDX - 1
        output = self.output_module.forward(tree.state[1], training)

        if len(outputs) > 0:
            # for index in range(len(outputs)):
            #     output = torch.add(output, outputs[index])
            # output = self.output_module.logsoftmax(output)

            if len(outputs) > 1:
                # print(tree.state[1].shape, output.shape)
                hidden = tree.state[1]
                print(output.transpose(0, 1).shape, hidden.shape)
                attn_weights = torch.bmm(output.transpose(0, 1), hidden)
                print(attn_weights.shape)
                exit(0)

        return output

    def get_child_states(self, tree):
        """
        Get c and h of all children
        :param tree:
        :return: (tuple)
        child_c: (num_children, 1, mem_dim)
        child_h: (num_children, 1, mem_dim)
        """
        if tree.num_children == 0:
            child_c = Var(torch.zeros(1, 1, self.mem_dim))
            child_h = Var(torch.zeros(1, 1, self.mem_dim))
        else:
            child_c = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            child_h = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            for idx in range(tree.num_children):
                child_c[idx] = tree.children[idx].state[0]
                child_h[idx] = tree.children[idx].state[1]
        return child_c, child_h

class Classifier(nn.Module):
    def __init__(self, mem_dim, num_classes, dropout=True):
        super(Classifier, self).__init__()
        self.mem_dim = mem_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.l1 = nn.Linear(self.mem_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, vec, training = False):
        if self.dropout:
            out = self.logsoftmax(self.l1(F.dropout(vec, training=training)))
        else:
            out = self.logsoftmax(self.l1(vec))
        return out

# putting the whole model together
class TreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, num_classes, criterion, vocab_output):
        super(TreeLSTM, self).__init__()
        self.tree_module = ChildSumTreeLSTM(in_dim, mem_dim, num_classes, criterion, vocab_output)
        self.classifier = Classifier(mem_dim, num_classes)
        self.tree_module.set_output_module(self.classifier)

    def forward(self, tree, inputs, training = False):
        output = self.tree_module(tree, inputs, training)
        return output
