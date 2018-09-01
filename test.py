import nltk

words = "VB PDT DT WP$ NNS POS NN VBZ NNP NNPS .".split()

freq = nltk.FreqDist(words)
print(freq["IN"] / len(words))
