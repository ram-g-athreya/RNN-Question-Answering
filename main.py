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

from fastText import load_model

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

    TOKS_INPUT_DIM = 300

    # get vocab object from vocab file previously written
    vocab_toks = Vocab(filename=os.path.join(args.data, 'vocab_toks.txt'), data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    vocab_pos = Vocab(filename=os.path.join(args.data, 'vocab_pos.txt'))
    vocab_rels = Vocab(filename=os.path.join(args.data, 'vocab_rels.txt'))

    vocab_output = Vocab(filename=os.path.join(args.data, 'vocab_output.txt'))

    # Set number of classes based on vocab_output
    args.num_classes = vocab_output.size()

    logger.debug('==> LC-QUAD vocabulary toks size : %d ' % vocab_toks.size())
    logger.debug('==> LC-QUAD vocabulary pos size : %d ' % vocab_pos.size())
    logger.debug('==> LC-QUAD vocabulary rels size : %d ' % vocab_rels.size())
    logger.debug('==> LC-QUAD output vocabulary size : %d ' % vocab_output.size())

    # load LC_QUAD dataset splits
    train_file = os.path.join(args.data, 'pth/lc_quad_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = LC_QUAD_Dataset(train_dir, vocab_toks, vocab_pos, vocab_rels, args.num_classes)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    test_file = os.path.join(args.data, 'pth/lc_quad_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = LC_QUAD_Dataset(test_dir, vocab_toks, vocab_pos, vocab_rels, args.num_classes)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    criterion = nn.NLLLoss()

    input_dim = vocab_pos.size() + vocab_rels.size() + TOKS_INPUT_DIM

    model = TreeLSTM(
        input_dim,
        args.mem_dim,
        args.num_classes,
        criterion,
        vocab_output
    )

    toks_embedding_model = nn.Embedding(vocab_toks.size(), TOKS_INPUT_DIM)
    pos_embedding_model = nn.Embedding(vocab_pos.size(), vocab_pos.size())
    rels_embedding_model = nn.Embedding(vocab_rels.size(), vocab_rels.size())

    pos_emb = torch.zeros(vocab_pos.size(), vocab_pos.size(), dtype=torch.float, device=device)
    rels_emb = torch.zeros(vocab_rels.size(), vocab_rels.size(), dtype=torch.float, device=device)

    for word in vocab_pos.labelToIdx.keys():
        # pos_emb[vocab_pos.getIndex(word)] = torch.Tensor(vocab_pos.size()).uniform_(-1, 1)
        word_index = vocab_pos.getIndex(word)
        word_vector = torch.zeros(1, vocab_pos.size())
        word_vector[0, word_index] = 1
        pos_emb[word_index] = word_vector

    for word in vocab_rels.labelToIdx.keys():
        # rels_emb[vocab_rels.getIndex(word)] = torch.Tensor(vocab_rels.size()).uniform_(-1, 1)
        word_index = vocab_rels.getIndex(word)
        word_vector = torch.zeros(1, vocab_rels.size())
        word_vector[0, word_index] = 1
        rels_emb[word_index] = word_vector

    toks_emb_file = os.path.join(args.data, 'pth/lc_quad_embed.pth')
    if os.path.isfile(toks_emb_file):
        toks_emb = torch.load(toks_emb_file)
    else:
        print("Generating FastText Word Vectors")
        toks_emb = torch.zeros(vocab_toks.size(), TOKS_INPUT_DIM, dtype=torch.float, device=device)
        fasttext_model = load_model("data/fasttext/wiki.en.bin")

        for word in vocab_toks.labelToIdx.keys():
            word_vector = fasttext_model.get_word_vector(word)
            if word_vector.all() != None and len(word_vector) == TOKS_INPUT_DIM:
                toks_emb[vocab_toks.getIndex(word)] = torch.Tensor(word_vector)
            else:
                toks_emb[vocab_toks.getIndex(word)] = torch.Tensor(TOKS_INPUT_DIM).uniform_(-1, 1)

        torch.save(toks_emb, toks_emb_file)

    # plug these into embedding matrix inside model
    toks_embedding_model.state_dict()['weight'].copy_(toks_emb)
    pos_embedding_model.state_dict()['weight'].copy_(pos_emb)
    rels_embedding_model.state_dict()['weight'].copy_(rels_emb)

    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad([
                {'params': model.parameters(), 'lr': args.lr}
            ], lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = Trainer(args, model, toks_embedding_model, pos_embedding_model, rels_embedding_model, criterion, optimizer, device, vocab_output)
    file_name = "analysis/expname={},input_dim={},mem_dim={},epochs={}".format(args.expname, input_dim, args.mem_dim, args.epochs)

    for epoch in range(args.epochs):
        print("\n" * 5)
        # Train Model
        train_loss = trainer.train(train_dataset)

        # Test Model on Training Dataset
        train_loss, train_pred = trainer.test(train_dataset)
        train_acc = metrics.accuracy_score(train_pred, train_dataset.labels, vocab_output)

        print('==> Train loss   : %f \t' % train_loss, end="")
        print('Epoch ', str(epoch + 1), 'train percentage ', train_acc)
        write_analysis_file(file_name, epoch, train_pred, train_dataset.labels, "train_acc", train_acc, train_loss, vocab_output)

        # Test Model on Testing Dataset
        test_loss, test_pred = trainer.test(test_dataset)
        test_acc = metrics.accuracy_score(test_pred, test_dataset.labels, vocab_output)

        print('==> Test loss   : %f \t' % test_loss, end="")
        print('Epoch ', str(epoch + 1), 'test percentage ', test_acc)
        write_analysis_file(file_name, epoch, test_pred, test_dataset.labels, "test_acc", test_acc, test_loss, vocab_output)

def write_analysis_file(file_name, epoch, predictions, labels, accuracy_label, accuracy, loss, vocab_output):
    with open(file_name + ",current_epoch={},{}={},loss={}.csv".format(epoch + 1, accuracy_label, accuracy ,loss), "w") as csv_file:
        writer = csv.writer(csv_file)
        preds = [vocab_output.getLabel(int(pred)) for pred in predictions]
        labels = labels.int().numpy()
        writer.writerows(zip(preds, labels))

if __name__ == "__main__":
    main()
