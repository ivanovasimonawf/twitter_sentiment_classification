from string_utils import tokenize_tweet, normalize_string
import random
from random import shuffle
from io import open
import pandas as pd
import numpy as np

data_file = '../data/kaggle_data/train.tsv'
test_file = '../data/kaggle_data/test_kaggle.tsv'
train_file = '../data/kaggle_data/train_kaggle.tsv'
test_file_glove = '../data/kaggle_data/test_kaggle_glove.tsv'
train_file_glove = '../data/kaggle_data/train_kaggle_glove.tsv'
glove_file = '../glove/glove.6B.300d.txt'

glove_words = []
f = open(glove_file, 'r')
for line in f:
    splitLine = line.split()
    glove_words.append(splitLine[0])


# split data into train and test
def split_data(data, train_split=0.8):
    random.seed(1)
    shuffle(data)
    split_index = int(train_split * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data


# save train and test data into separate files
def save_split_data(data, path):
    file = open(path, 'w')
    for line in data:
        if len(line) != 2:
            continue
        file.write(line[0].strip() + '\t' + line[1].strip() + '\n')


# read original .csv file and save it as .tsv
def save_as_tsv(filename):
    dataframe = pd.read_csv(filename)[['Sentiment', 'SentimentText']]
    dataframe.to_csv('../data/kaggle_data/train.tsv', sep='\t')


def avg_words(data):
    number_of_words = 0
    for line in data:
        if len(line) != 2:
            continue
        number_of_words += len(line[1].split(' '))
    return float(number_of_words)/len(data)


def get_glove_words_only(sentence):
    new_sentence = ""
    for word in sentence.split():
        if word.strip() in glove_words:
           new_sentence += word + ' '
    return new_sentence


lines = open(data_file, encoding='ISO-8859-1').read().strip().split('\n')
sentiment_tweet_pairs = [[tokenize_tweet(normalize_string(s)) for s in l.split('\t')][-2:] for l in lines][1:]


def extract_glove(data):
    new_pairs = []
    for pair in data:
        if len(pair) != 2:
            continue
        new_pairs.append([pair[0], get_glove_words_only(pair[1])])
    return new_pairs


train_data, test_data = split_data(sentiment_tweet_pairs)
train_data_glove, test_data_glove = extract_glove(train_data), extract_glove(test_data)
print("Train size: ", len(train_data))
print("Test size: ", len(test_data))
print("Average word count train: ", avg_words(train_data))
print("Average word count test: ", avg_words(test_data))

save_split_data(train_data, train_file)
save_split_data(test_data, test_file)

save_split_data(train_data_glove, train_file_glove)
save_split_data(test_data_glove, test_file_glove)
