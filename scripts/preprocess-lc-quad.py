"""
Preprocessing script for LC-QUAD data.
"""

import glob
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

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


def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
           % (cp, tokpath, parentpath, tokenize_flag, filepath))
    os.system(cmd)


def build_vocab(filepaths, dst_path, lowercase=False):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def split(filepath, dst_dir):
    with open(filepath) as datafile, \
            open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'input.txt'), 'w') as inputfile, \
            open(os.path.join(dst_dir, 'output.txt'), 'w') as outputfile:

        data = json.load(datafile)
        for datum in data:
            idfile.write(datum["_id"] + "\n")
            inputfile.write(datum["corrected_question"] + "\n")
            outputfile.write(str(datum["sparql_template_id"]) + "\n")


def split_data(X, y, dst_dir):
    with open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'input.txt'), 'w') as inputfile, \
            open(os.path.join(dst_dir, 'output.txt'), 'w') as outputfile:
        y = y.tolist()

        for index in range(len(X)):
            corrected_question = str(X.iloc[index]["corrected_question"])
            # corrected_question = str(X.iloc[index]["corrected_question"]).replace('(', '').replace(')', '')
            idfile.write(str(X.iloc[index]["_id"]) + "\n")
            inputfile.write(corrected_question + "\n")
            outputfile.write(str(y[index]) + "\n")


def parse(dirpath, cp=''):
    dependency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)
    # constituency_parse(os.path.join(dirpath, 'input.txt'), cp=cp, tokenize=True)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing LC-QUAD dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lc_quad_dir = os.path.join(data_dir, 'lc-quad')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(lc_quad_dir, 'train')
    test_dir = os.path.join(lc_quad_dir, 'test')
    make_dirs([train_dir, test_dir])
    make_dirs([os.path.join(lc_quad_dir, 'pth')])

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.9.1-models.jar')])

    # Load Data
    df_train = pd.read_json(os.path.join(lc_quad_dir, "train-data.json"))
    df_test = pd.read_json(os.path.join(lc_quad_dir, "test-data.json"))
    df = pd.concat([df_train, df_test], ignore_index = True)

    X = df.loc[:, df.columns != 'sparql_template_id']
    y = df['sparql_template_id'].tolist()

    for index in range(len(y)):
        id = y[index]
        if id >= 300 and id < 500: # Collapse 3xx or 4xx templates to their corresponding template - 300 id since all that differentiates them is extra rdf:type class triple which can be added as an optional triple to SPARQL Query
            y[index] = id - 300
        elif id == 152: # Effectively Template 152 and Template 151 are the same template or can be converted into single SPARQL Query
            y[index] = 151

        # Because they have very similar but inverted structures
        if y[index] == 2:
            y[index] = 1
        elif y[index] == 106:
            y[index] = 105
        elif y[index] == 5:
            y[index] = 3

    y = pd.Series(y)
    print(y.value_counts(normalize=True))

    df_combined = X
    df_combined['sparql_template_id'] = y
    df.to_csv(os.path.join(lc_quad_dir, 'dataset.csv'), index=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    split_data(X_train, y_train, train_dir)
    split_data(X_test, y_test, test_dir)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # Build Vocabulary for input
    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/*.pos')), # All POS and RELS files
        os.path.join(lc_quad_dir, 'vocab_pos.txt'))

    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/*.rels')),
        # All POS and RELS files
        os.path.join(lc_quad_dir, 'vocab_rels.txt'))

    # Build Vocabulary for output
    build_vocab(
        glob.glob(os.path.join(lc_quad_dir, '*/output.txt')),
        os.path.join(lc_quad_dir, 'vocab_output.txt'))
