from copy import deepcopy

import torch


class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def accuracy_score(self, predictions, labels, vocab_output):
        labels = torch.tensor([vocab_output.getIndex(str(int(label))) for label in labels], dtype=torch.float)
        correct = (predictions == labels).sum()
        total = labels.size(0)
        acc = float(correct) / total
        return acc