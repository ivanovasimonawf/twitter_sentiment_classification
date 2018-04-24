import numpy as np

class TwitterSentimentCorpus:
    def __init__(self):
        self.word2vec = {}
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 2
        self.model = {}


    def add_sentence(self, sentence):
        known_words = 0
        for word in sentence.split():
            if self.add_word(word):
                known_words += 1
        return True if known_words != 0 else False


    def add_to_word2vec(self, word):
        known_word = 0
        try:
            self.word2vec[word] = self.model[word]
        except:
            known_word = 1
        return known_word


    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            known_word = self.add_to_word2vec(word)
        else:
            self.word2count[word] += 1
            known_word = self.add_to_word2vec(word)
        return True if known_word == 0 else False

    def create_word2vec_dictionary(self, glove_file):
        f = open(glove_file, 'r')
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            self.model[word] = embedding
