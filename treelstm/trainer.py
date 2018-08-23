from tqdm import tqdm

import torch
import torch.nn as nn

from . import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device, vocab_output):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.vocab_output = vocab_output
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')

        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, input, label = dataset[indices[idx]]
            target = utils.map_label_to_target(label, dataset.num_classes, self.vocab_output)

            input = input.to(self.device)
            target = target.to(self.device)

            # output = self.model(ltree, linput, rtree, rinput)
            output = self.model(tree, input)
            m = nn.LogSoftmax()
            _, pred = torch.max(output.data, 1)

            loss = self.criterion(m(output), target)
            total_loss += loss.item()
            loss.backward()

            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.step() # Need to check how to use this properly. For now doing what is specified everywhere
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')

            indices = torch.arange(0, dataset.num_classes, dtype=torch.float, device='cpu') # THIS LINE WAS CHANGED FROM 1 to num_classes + 1 to 0 and num_classes
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                tree, input, label = dataset[idx]
                target = utils.map_label_to_target(label, dataset.num_classes, self.vocab_output)

                input = input.to(self.device)
                target = target.to(self.device)

                output = self.model(tree, input)
                m = nn.LogSoftmax()
                _, pred = torch.max(output.data, 1)
                loss = self.criterion(m(output), target)
                total_loss += loss.item()
                predictions[idx] = pred
                # predictions[idx] = torch.dot(indices, torch.exp(output))
        return total_loss / len(dataset), predictions
