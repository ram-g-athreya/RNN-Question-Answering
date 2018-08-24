from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable as Var
from . import utils
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, args, model, embedding_model, criterion, optimizer, device, vocab_output):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.embedding_model = embedding_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.vocab_output = vocab_output
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.embedding_model.train()
        self.embedding_model.zero_grad()
        self.optimizer.zero_grad()
        total_loss, k = 0.0, 0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            tree, input, label = dataset[indices[idx]]
            input = Var(input)
            target = Var(utils.map_label_to_target(label, dataset.num_classes, self.vocab_output))

            input = input.to(self.device)
            target = target.to(self.device)

            emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target)

            err = err / self.args.batchsize
            total_loss += err.data[0]
            err.backward()
            k += 1

            if k == self.args.batchsize:
                for f in self.embedding_model.parameters():
                    f.data.sub_(f.grad.data * self.args.emblr)
                self.optimizer.step()
                self.embedding_model.zero_grad()
                self.optimizer.zero_grad()
                k = 0
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        self.embedding_model.eval()
        total_loss = 0
        predictions = torch.zeros(len(dataset))
        indices = torch.range(0, dataset.num_classes) # THIS LINE WAS CHANGED FROM 1 to num_classes + 1 to 0 and num_classes

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            tree, sent, label = dataset[idx]
            input = Var(sent, volatile=True)
            target = utils.map_label_to_target(label, dataset.num_classes, self.vocab_output)

            input = input.to(self.device)
            target = target.to(self.device)

            emb = F.torch.unsqueeze(self.embedding_model(input), 1)
            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target)
            total_loss += err.data[0]

            val, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]

            # output = self.model(tree, input)
            # m = nn.LogSoftmax()
            # _, pred = torch.max(output.data, 1)
            # loss = self.criterion(m(output), target)
            # total_loss += loss.item()
            # predictions[idx] = pred
        return total_loss / len(dataset), predictions
