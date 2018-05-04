from string_utils import tokenize_tweet, normalize_string
import string
import random
from random import shuffle
from io import open
import pandas as pd
import operator

# original train data file
data_file = '../data/kaggle_data/train.tsv'
# path where to save separated train and test without preprocessing (except user, link, hashtag)
test_file = '../data/punctuation/test_kaggle_full.tsv'
train_file = '../data/punctuation/train_kaggle_full.tsv'
# path where to save tweets with words that we have glove vectors for
test_file_glove = '../data/punctuation/test_kaggle_glove.tsv'
train_file_glove = '../data/punctuation/train_kaggle_glove.tsv'
# path where to save tweets with removed one-time words
test_file_reduced = '../data/punctuation/test_kaggle_reduced.tsv'
train_file_reduced = '../data/punctuation/train_kaggle_reduced.tsv'
glove_file = '../glove/glove.6B.300d.txt'

glove_words = []
word2count = {}
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


# save data into file line by line (tab-separated label, tweet)
def save_split_data(data, path):
    file = open(path, 'w')
    for line in data:
        if len(line) != 2:
            continue
        file.write(line[0].strip() + '\t' + line[1].strip() + '\n')


# read original .csv file and save it as .tsv (only used once because i couldn't read the file as csv because of commas
# inside tweets)
def save_as_tsv(filename):
    dataframe = pd.read_csv(filename)[['Sentiment', 'SentimentText']]
    dataframe.to_csv('../data/kaggle_data/train.tsv', sep='\t')


# average number of words per tweet in a dataset
def avg_words(data):
    number_of_words = 0
    for line in data:
        if len(line) != 2:
            continue
        number_of_words += len(line[1].split(' '))
    return float(number_of_words)/len(data)


# leave only glove-available words in a sentence
def get_glove_words_only(sentence):
    new_sentence = ""
    for word in sentence.split():
        if word.strip() in glove_words:
           new_sentence += word + ' '
        if word.strip() not in word2count:
            word2count[word.strip()] = 1
        else:
            word2count[word.strip()] += 1
    return new_sentence


# extract sentences that will have 0 length because there are no glove vectors available for the words
# returns label, tweet pairs with length different of 0
def extract_glove(data):
    new_pairs = []
    for pair in data:
        if len(pair) != 2:
            continue
        extracted_sentence = get_glove_words_only(pair[1])
        if len(extracted_sentence.split()) != 0:
            new_pairs.append([pair[0], extracted_sentence])
    return new_pairs


# leave words in a sentence repeated more than once in whole dataset
def extract_one_time_words(sentence):
    new_sentence = ""
    for word in sentence.split():
        if word2count[word.strip()] != 1:
            new_sentence += word + ' '
    return new_sentence


# extract sentences that will have 0 length because we remove words that are only repeated once
# returns label, tweet pairs with length different of 0
def extract_sentences(data):
    new_pairs = []
    for pair in data:
        extracted_sentence = extract_one_time_words(pair[1])
        if len(extracted_sentence.split()) != 0:
            new_pairs.append([pair[0], extracted_sentence])
    return new_pairs


def extract_punctuation(sentence):
    words_in_sentence = sentence.split()
    new_sentence_words = [words_in_sentence[0]]
    for i in range(1, len(words_in_sentence)):
        current_word = words_in_sentence[i].strip()
        if current_word not in string.punctuation:
            new_sentence_words.append(current_word)
        else:
            if current_word != new_sentence_words[-1]:
                new_sentence_words.append(current_word)
    new_sentence = ""
    for word in new_sentence_words:
        new_sentence += word + " "
    return new_sentence


def extract_punctuations(data):
    new_pairs = []
    for pair in data:
        extracted_sentence = extract_punctuation(pair[1])
        if len(extracted_sentence.split()) != 0:
            new_pairs.append([pair[0], extracted_sentence])
    return new_pairs


# prints first print_first words with most repetitions
def print_repetitions(print_first = 100):
    sorted_words = sorted(word2count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_words[:print_first])


lines = open(data_file, encoding='ISO-8859-1').read().strip().split('\n')
sentiment_tweet_pairs = [[tokenize_tweet(normalize_string(s)) for s in l.split('\t')][-2:] for l in lines][1:]

train_data, test_data = split_data(sentiment_tweet_pairs)
train_data_glove, test_data_glove = extract_glove(train_data), extract_glove(test_data)
save_split_data(train_data_glove, train_file_glove)
save_split_data(test_data_glove, test_file_glove)
train_data_glove, test_data_glove = extract_punctuations(extract_sentences(train_data_glove)), extract_punctuations(extract_sentences(test_data_glove))
print("Train size full: ", len(train_data))
print("Test size full: ", len(test_data))
print("Train size reduced: ", len(train_data_glove))
print("Test size reduced: ", len(test_data_glove))
print("Average word count train: ", avg_words(train_data))
print("Average word count test: ", avg_words(test_data))

save_split_data(train_data, train_file)
save_split_data(test_data, test_file)

save_split_data(train_data_glove, train_file_reduced)
save_split_data(test_data_glove, test_file_reduced)

print_repetitions()
