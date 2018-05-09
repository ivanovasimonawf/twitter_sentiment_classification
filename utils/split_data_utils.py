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
test_file = '../data/kaggle_data/test_kaggle_full.tsv'
train_file = '../data/kaggle_data/train_kaggle_full.tsv'
# path where to save tweets with removed one-time words
test_file_reduced = '../data/kaggle_data/test_kaggle_reduced.tsv'
train_file_reduced = '../data/kaggle_data/train_kaggle_reduced.tsv'
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
    return float(number_of_words) / len(data)


def create_word2vec(data):
    for pair in data:
        if len(pair) != 2:
            continue
        for word in pair[1].split():
            if word.strip() not in word2count:
                word2count[word.strip()] = 1
            else:
                word2count[word.strip()] += 1


def preprocess_data(data, remove_glove=True, remove_one_time=True, remove_repeatable_punc=True):
    new_pairs = []
    glove_pairs = []
    for pair in data:
        if len(pair) != 2:
            continue
        sentence = pair[1]
        if remove_glove:
            sentence = get_glove_words_only(sentence)
        if is_len_zero(sentence):
            continue
        else:
            glove_pairs.append([pair[0], sentence])
        if remove_one_time:
            sentence = extract_one_time_words(sentence)
        if is_len_zero(sentence):
            continue
        if remove_repeatable_punc:
            sentence = extract_punctuation(sentence)
        if not is_len_zero(sentence):
            new_pairs.append([pair[0], sentence])
    return new_pairs, glove_pairs


def is_len_zero(sentence):
    if len(sentence.split()) == 0:
        return True
    else:
        return False


# leave only glove-available words in a sentence
def get_glove_words_only(sentence):
    new_sentence = ""
    for word in sentence.split():
        if word.strip() in glove_words:
            new_sentence += word.strip() + ' '
    return new_sentence


# leave words in a sentence repeated more than once in whole dataset
def extract_one_time_words(sentence):
    new_sentence = ""
    for word in sentence.split():
        if word2count[word.strip()] != 1:
            new_sentence += word + ' '
    return new_sentence


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


# prints first print_first words with most repetitions
def print_repetitions(print_first=100):
    sorted_words = sorted(word2count.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_words[:print_first])


lines = open(data_file, encoding='ISO-8859-1').read().strip().split('\n')
sentiment_tweet_pairs = [[tokenize_tweet(normalize_string(s)) for s in l.split('\t')][-2:] for l in lines][1:]

train_data, test_data = split_data(sentiment_tweet_pairs)
create_word2vec(train_data)
create_word2vec(test_data)
train_data_preprocessed, train_glove_only = preprocess_data(train_data)
test_data_preprocessed, test_glove_only = preprocess_data(test_data)
print("Train size full: ", len(train_data))
print("Test size full: ", len(test_data))
print("Train size reduced: ", len(train_data_preprocessed))
print("Test size reduced: ", len(test_data_preprocessed))
print("Average word count train: ", avg_words(train_data))
print("Average word count test: ", avg_words(test_data))

save_split_data(train_glove_only, train_file)
save_split_data(test_glove_only, test_file)

save_split_data(train_data_preprocessed, train_file_reduced)
save_split_data(test_data_preprocessed, test_file_reduced)

print_repetitions()
