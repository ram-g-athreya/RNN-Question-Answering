from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

import csv

# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import TreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR LC_QUAD DATASET
from treelstm import LC_QUAD_Dataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args


# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train/')
    test_dir = os.path.join(args.data, 'test/')

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=os.path.join(args.data, 'vocab.txt'),
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])

    vocab_output = Vocab(filename=os.path.join(args.data, 'vocab_output.txt'))
    logger.debug('==> LC-QUAD vocabulary size : %d ' % vocab.size())
    logger.debug('==> LC-QUAD output vocabulary size : %d ' % vocab_output.size())

    # load LC_QUAD dataset splits
    train_file = os.path.join(args.data, 'pth/lc_quad_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = LC_QUAD_Dataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    test_file = os.path.join(args.data, 'pth/lc_quad_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = LC_QUAD_Dataset(test_dir, vocab, args.num_classes)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    criterion = nn.NLLLoss()
    # initialize model, criterion/loss_function, optimizer
    model = TreeLSTM(
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.num_classes,
        criterion,
        vocab_output
    )

    embedding_model = nn.Embedding(vocab.size(), args.input_dim)
    emb_file = os.path.join(args.data, 'pth/lc_quad_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)

        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
            else:
                emb[vocab.getIndex(word)] = torch.Tensor(emb[vocab.getIndex(word)].size()).normal_(-0.05, 0.05)

        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    # model.emb.weight.data.copy_(emb)
    embedding_model.state_dict()['weight'].copy_(emb)

    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = Trainer(args, model, embedding_model, criterion, optimizer, device, vocab_output)

    best = -float('inf')
    file_name = "analysis/input_dim={},mem_dim={},epochs={}".format(args.input_dim, args.mem_dim, args.epochs)

    for epoch in range(args.epochs):
        print("\n" * 5)
        # Train Model
        train_loss = trainer.train(train_dataset)

        # Test Model on Training Dataset
        train_loss, train_pred = trainer.test(train_dataset)
        train_acc = metrics.accuracy_score(train_pred, train_dataset.labels, vocab_output)

        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ', str(epoch + 1), 'train percentage ', train_acc)
        write_analysis_file(file_name, epoch, train_pred, train_dataset.labels, "train_acc", train_acc, vocab_output)

        # Test Model on Testing Dataset
        test_loss, test_pred = trainer.test(test_dataset)
        test_acc = metrics.accuracy_score(test_pred, test_dataset.labels, vocab_output)

        print('==> Test loss   : %f \t' % test_loss, end="")
        print('Epoch ', str(epoch + 1), 'test percentage ', test_acc)
        write_analysis_file(file_name, epoch, test_pred, test_dataset.labels, "test_acc", test_acc, vocab_output)

def write_analysis_file(file_name, epoch, predictions, labels, accuracy_label, accuracy, vocab_output):
    with open(file_name + ",current_epoch={},{}={}.csv".format(epoch + 1, accuracy_label, accuracy), "w") as csv_file:
        writer = csv.writer(csv_file)
        preds = [vocab_output.getLabel(int(pred)) for pred in predictions]
        labels = labels.int().numpy()
        writer.writerows(zip(preds, labels))

if __name__ == "__main__":
    main()
