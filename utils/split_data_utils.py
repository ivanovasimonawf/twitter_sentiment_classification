from string_utils import tokenize_tweet, normalize_string
import random
from random import shuffle
from io import open
import pandas as pd

data_file = '../data/kaggle_data/train.tsv'
test_file = '../data/kaggle_data/test_kaggle.tsv'
train_file = '../data/kaggle_data/train_kaggle.tsv'


# split data into train and test
def split_data(train_split=0.8):
    random.seed(1)
    shuffle(sentiment_tweet_pairs)
    split_index = int(train_split * len(sentiment_tweet_pairs))
    train_data = sentiment_tweet_pairs[:split_index]
    test_data = sentiment_tweet_pairs[split_index:]
    return train_data, test_data


# save train and test data into separate files
def save_split_data(data, path):
    file = open(path, 'w')
    for line in data:
        if len(line) != 2:
            continue
        file.write(line[0] + '\t' + line[1] + '\n')


# read original .csv file and save it as .tsv
def save_as_tsv(filename):
    dataframe = pd.read_csv(filename)[['Sentiment', 'SentimentText']]
    dataframe.to_csv('../data/kaggle_data/train.tsv', sep='\t')


lines = open(data_file, encoding='ISO-8859-1').read().strip().split('\n')
sentiment_tweet_pairs = [[tokenize_tweet(normalize_string(s)).strip() for s in l.split('\t')][-2:] for l in lines][1:]

train_data, test_data = split_data()
print("Train size: ", len(train_data))
print("Test size: ", len(test_data))
save_split_data(train_data, train_file)
save_split_data(test_data, test_file)
