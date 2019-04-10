# Template-based Question Answering over Recursive Neural Networks
This project contains the source code for a template classification model on the LC-QuAD dataset using recursive neural network implemented in Pytorch.

## Instructions
First download the LC-QuAD dataset and pre-process it into train and test data.
```
$bash scripts/download.sh # Download LC-QuAD
$bash scripts/preprocess-lc-quad.sh # Pre-process LC-QuAD Dataset into train, test split
```

Download Facebook FastText which is used as the embedding model:
```
mkdir data
$wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip # Download FastText
$unzip wiki.en.zip
$mv wiki.en.bin data/
$rm wiki.en.zip
```

Generating the model using Pytorch:
```
$python main.py --epochs=7 # Further configuration options can be found in config.py
```
