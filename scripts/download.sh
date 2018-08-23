#!/bin/bash
set -e

echo "Download LC-QUAD Dataset"
mkdir -p data/lc-quad
cd data/lc-quad/
wget -q -c https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/train-data.json
wget -q -c https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/test-data.json
cd ../../

echo "Downloading Stanford parser and tagger"
cd lib/
wget -q -c http://nlp.stanford.edu/software/stanford-postagger-2015-01-29.zip
unzip -q stanford-postagger-2015-01-29.zip
mv stanford-postagger-2015-01-30/ stanford-tagger
rm stanford-postagger-2015-01-29.zip

wget -q -c http://nlp.stanford.edu/software/stanford-parser-full-2015-01-29.zip
unzip -q stanford-parser-full-2015-01-29.zip
mv stanford-parser-full-2015-01-30/ stanford-parser
rm stanford-parser-full-2015-01-29.zip
cd ../