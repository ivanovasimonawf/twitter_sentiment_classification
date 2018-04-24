import numpy as np
from io import open
from twitter_corpus import TwitterSentimentCorpus

train_file = 'data/kaggle_data/train_kaggle.tsv'
test_file =  'data/kaggle_data/test_kaggle.tsv'
glove_file = 'glove/glove.6B.300d.txt'

all_categories = []
train_data_lines = []
test_data_lines = []
twitter_corpus = TwitterSentimentCorpus()


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


# remove tweets with len 0
def remove_tweets_at_indexes(tweets, labels, indexes):
    sub = 0
    for index in indexes:
        index -= sub
        del tweets[index]
        del labels[index]
        sub += 1
    return tweets, labels


# add all words from tweets in word2vec dictionary and remove any tweets that have len 0 (no glove vectos is available for them)
def fill_twitter_corpus(tweets, labels):
    indexes_to_remove = []
    for index in range(len(tweets)):
        if labels[index] not in all_categories:
            all_categories.append(labels[index])
        if add_tweet_to_corpus(tweets[index]) == False:
            indexes_to_remove.append(index)
    tweets, labels = remove_tweets_at_indexes(tweets, labels, indexes_to_remove)
    return tweets, labels


# add words from a tweet in word2vec dictionary (returns True if no words were available as glove vectors)
def add_tweet_to_corpus(tweet):
    return twitter_corpus.add_sentence(tweet)


# read train/test data and prepare the twitter corpus
def prepare_data():
    print('Prepare data')
    global train_data_lines, test_data_lines
    twitter_corpus.create_word2vec_dictionary(glove_file)
    train_tweets, train_labels = \
        separate_data_labels([line.split('\t') for line in open(train_file, encoding='utf-8').read().strip().split('\n')][1:])
    test_tweets, test_labels = \
        separate_data_labels([line.split('\t') for line in open(test_file, encoding='utf-8').read().strip().split('\n')][1:])
    train_tweets, train_labels = fill_twitter_corpus(train_tweets, train_labels)
    train_data_lines = [[label, tweet] for tweet, label in zip(train_tweets, train_labels)]
    test_tweets, test_labels = fill_twitter_corpus(test_tweets, test_labels)
    test_data_lines = [[label, tweet] for tweet, label in zip(test_tweets, test_labels)]
    twitter_corpus.model = {}

prepare_data()
print("Data Prepared.")
