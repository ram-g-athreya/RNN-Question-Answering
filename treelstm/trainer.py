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
            tree, pos_sent, rels_sent, label = dataset[indices[idx]]
            pos_sent = Var(pos_sent)
            rels_sent = Var(rels_sent)

            target = Var(utils.map_label_to_target(label, dataset.num_classes, self.vocab_output))

            pos_sent = pos_sent.to(self.device)
            rels_sent = rels_sent.to(self.device)
            target = target.to(self.device)

            pos_emb = F.torch.unsqueeze(self.embedding_model(pos_sent), 1)
            rels_emb = F.torch.unsqueeze(self.embedding_model(rels_sent), 1)
            emb = torch.cat((pos_emb, rels_emb), 2)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target)

            err = err / self.args.batchsize
            total_loss += err.item()
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

        for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
            torch.no_grad()
            tree, pos_sent, rels_sent, label = dataset[idx]
            pos_sent = Var(pos_sent)
            rels_sent = Var(rels_sent)

            target = utils.map_label_to_target(label, dataset.num_classes, self.vocab_output)

            pos_sent = pos_sent.to(self.device)
            rels_sent = rels_sent.to(self.device)
            target = target.to(self.device)

            pos_emb = F.torch.unsqueeze(self.embedding_model(pos_sent), 1)
            rels_emb = F.torch.unsqueeze(self.embedding_model(rels_sent), 1)
            emb = torch.cat((pos_emb, rels_emb), 2)

            output = self.model.forward(tree, emb, training=True)
            err = self.criterion(output, target)
            total_loss += err.item()

            _, pred = torch.max(output, 1)

            predictions[idx] = pred.data.cpu()[0]
        return total_loss / len(dataset), predictions
