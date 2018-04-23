import numpy as np
from io import open
from twitter_corpus import TwitterSentimentCorpus

train_file = 'data/split_data/train_data1.tsv'
test_file = 'data/split_data/test_data1.tsv'
# za kaggle start
# train_file = 'data/kaggle_data/train_kaggle.tsv'
# test_file =  'data/kaggle_data/test_kaggle.tsv'
# za kaggle end
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
        # za kaggle start
        # if dataset[i][0] != '0 ' and dataset[i][0] != '1 ':
        #     continue
        # za kaggle end
        labels.append(dataset[i][0])
        data.append(dataset[i][1])
    return data, labels


# add all words in the word2vec dictionary
def fill_twitter_corpus(tweets, labels):
    for tweet, label in zip(tweets, labels):
        if label not in all_categories:
            all_categories.append(label)
        twitter_corpus.add_sentence(tweet)
    return twitter_corpus


# read train/test data and prepare the twitter corpus
def prepare_data():
    global train_data_lines, test_data_lines
    train_data_lines= [line.split('\t') for line in open(train_file, encoding='utf-8').read().strip().split('\n')]
    test_data_lines = [line.split('\t') for line in open(test_file, encoding='utf-8').read().strip().split('\n')]
    # za kaggle start
    # train_data_lines = [line.split('\t') for line in open(train_file, encoding='utf-8').read().strip().split('\n')][1:]
    # test_data_lines = [line.split('\t') for line in open(test_file, encoding='utf-8').read().strip().split('\n')][1:]
    # za kaggle end
    train_tweets, train_labels = separate_data_labels(train_data_lines)
    test_tweets, test_labels = separate_data_labels(test_data_lines)
    fill_twitter_corpus(train_tweets, train_labels)
    fill_twitter_corpus(test_tweets, test_labels)
    twitter_corpus.create_word2vec_dictionary(glove_file)

prepare_data()
print("Data Prepared.")
