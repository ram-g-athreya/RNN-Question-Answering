#!/bin/bash
set -e

echo "Download LC-QUAD Dataset"
mkdir -p data/lc-quad
cd data/lc-quad/
wget -q -c https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/train-data.json
wget -q -c https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/test-data.json

echo "Download LC-QUAD Template Metadata"
wget -q -c https://raw.githubusercontent.com/AskNowQA/LC-QuAD/data/resources/templates.json
cd ../../

echo "Downloading Stanford parser and tagger"
cd lib/
wget -q -c https://nlp.stanford.edu/software/stanford-postagger-2018-02-27.zip
unzip -q stanford-postagger-2018-02-27.zip
mv stanford-postagger-2018-02-27/ stanford-tagger
rm stanford-postagger-2018-02-27.zip

wget -q -c https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip
unzip -q stanford-parser-full-2018-02-27.zip
mv stanford-parser-full-2018-02-27/ stanford-parser
rm stanford-parser-full-2018-02-27.zip
cd ../