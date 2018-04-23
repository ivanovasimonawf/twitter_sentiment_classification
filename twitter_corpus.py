import numpy as np

class TwitterSentimentCorpus:
    def __init__(self):
        self.word2vec = {}
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 2

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def create_word2vec_dictionary(self, glove_file):
        print("Creating word2vec dictionary...")
        f = open(glove_file, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        for word in self.word2index:
            try:
                self.word2vec[word] = model[word]
            except:
                continue
