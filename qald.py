import os
import torch
import json

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as Var
from treelstm import Vocab
from treelstm import LC_QUAD_Dataset
from treelstm import Constants

from main import generate_embeddings

def dependency_parse(filepath, cp='', tokenize=True):
    print('\nDependency parsing ' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath = os.path.join(dirpath, filepre + '.rels')
    pospath = os.path.join(dirpath, filepre + '.pos')
    lenpath = os.path.join(dirpath, filepre + '.len')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s -pospath %s -lenpath %s %s < %s'
           % (cp, tokpath, parentpath, relpath, pospath, lenpath, tokenize_flag, filepath))
    print(cmd)
    os.system(cmd)

if __name__ == '__main__':
    lib_dir = 'lib/'
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.9.1-models.jar')])

    data = pd.read_csv('data/qald/qald-test.csv')
    questions = data['question']

    with open('data/qald/input.txt', 'w') as inputfile:
        for index in range(len(questions)):
            question = questions[index]
            inputfile.write(question + "\n")

    dependency_parse('data/qald/input.txt', cp=classpath)

    saved_model = torch.load('checkpoints/Down to 15 templates higher dropout,epoch=5,test_acc=0.8205394190871369.pt')
    trainer = saved_model['trainer']

    vocab_rels = Vocab(filename='data/lc-quad/vocab_rels.txt')
    vocab_pos = Vocab(filename='data/lc-quad/vocab_pos.txt')
    vocab_toks = Vocab(filename='data/lc-quad/vocab_toks.txt', data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    vocab_output = trainer.vocabs['output']

    toks_emb = generate_embeddings(vocab_toks, 'data/lc-quad/pth/lc_quad_toks_embed.pth')
    toks_embedding_model = nn.Embedding(vocab_toks.size(), 300)
    toks_embedding_model.state_dict()['weight'].copy_(toks_emb)
    trainer.embeddings['toks'] = toks_embedding_model
    trainer.vocabs['toks'] = vocab_toks

    train_dataset = LC_QUAD_Dataset('data/qald', vocab_toks, vocab_pos, vocab_rels, 0)

    json_object = []
    for index in range(len(train_dataset)):
        tree, toks_sent, pos_sent, rels_sent, label = train_dataset[index]
        toks_sent = Var(toks_sent)
        pos_sent = Var(pos_sent)
        rels_sent = Var(rels_sent)

        toks_emb = F.torch.unsqueeze(trainer.embeddings['toks'](toks_sent), 1)
        pos_emb = F.torch.unsqueeze(trainer.embeddings['pos'](pos_sent), 1)
        rels_emb = F.torch.unsqueeze(trainer.embeddings['rels'](rels_sent), 1)
        chars_emb = trainer.get_char_vector(toks_sent)
        emb = torch.cat((toks_emb, pos_emb, rels_emb, chars_emb), 2)

        output = trainer.model.forward(tree, emb, training=False)
        _, pred = torch.topk(output[0].squeeze(0), 2)
        pred = pred.numpy()

        json_object.append({
            'question': questions[index],
            'predictions': str(vocab_output.getLabel(pred[0])) + ',' + str(vocab_output.getLabel(pred[1])),
            'actual': str(data.loc[index, 'template'])
        })

    with open('data/qald/qald.json', 'w') as outfile:
        json.dump(json_object, outfile)