import numpy as np
from io import open
from twitter_corpus import TwitterSentimentCorpus

option = 'full'

train_file = 'data/punctuation/train_kaggle_' + option + '.tsv'
test_file =  'data/punctuation/test_kaggle_' + option + '.tsv'
glove_file = 'glove/glove.6B.300d.txt'

all_categories = []
train_data_lines = []
test_data_lines = []
twitter_corpus = TwitterSentimentCorpus()


print(train_file)
print(test_file)

# separate data into tweets and labels
def separate_data_labels(dataset):
    data = []
    labels = []
    for i in range(len(dataset)):
        if dataset[i][0] != '0' and dataset[i][0] != '1':
            continue
        labels.append(dataset[i][0])
        data.append(dataset[i][1])
    return data, labels


# add all words from tweets in word2vec dictionary and remove any tweets that have len 0 (no glove vectos is available for them)
def fill_twitter_corpus(tweets, labels):
    for index in range(len(tweets)):
        if labels[index] not in all_categories:
            all_categories.append(labels[index])
        twitter_corpus.add_sentence(tweets[index])


# read train/test data and prepare the twitter corpus
def prepare_data():
    print('Prepare data')
    global train_data_lines, test_data_lines
    twitter_corpus.create_word2vec_dictionary(glove_file)
    train_tweets, train_labels = \
        separate_data_labels([line.split('\t') for line in open(train_file, encoding='utf-8').read().strip().split('\n')])
    test_tweets, test_labels = \
        separate_data_labels([line.split('\t') for line in open(test_file, encoding='utf-8').read().strip().split('\n')])
    fill_twitter_corpus(train_tweets, train_labels)
    fill_twitter_corpus(test_tweets, test_labels)
    train_data_lines = [[label, tweet] for tweet, label in zip(train_tweets, train_labels)]
    test_data_lines = [[label, tweet] for tweet, label in zip(test_tweets, test_labels)]
    twitter_corpus.model = {}

prepare_data()
print("Data Prepared.")