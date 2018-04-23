import unicodedata
import re

user_token = "user"
hashtag_token = "hashtag"
url_token = "link"


# convert to ascii and remove numbers
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", s)
    return s


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# separate punctuation connected with a word
def extract_punctuation(word):
    punctuations = ". , ; : 's ! n't 'm * ?"
    for punctuation in punctuations.split(' '):
        if punctuation in word:
            if punctuation == '.':
                punctuation = '\.'
            elif punctuation == '*':
                punctuation = '\*'
            elif punctuation == '?':
                punctuation = '\?'
            return list(filter(None, re.split('(' + punctuation + ')', word)))
    return [word]


# split sequence into words for which i need to use glove vectors
def split_sequence(sequence):
    words_in_sequence = sequence.split(' ')
    all_words_for_vecs = []
    new_sequence = ""
    for word in words_in_sequence:
        extracted = extract_punctuation(word)
        for ext in extracted:
            all_words_for_vecs.append(ext)
            new_sequence += ext + " "
    return new_sequence


# replace #word with "hashtag" + word, @name with "user", and urls with "link"
def tokenize_tweet(input):
    input = input.lower()
    input = re.sub('@\w+', user_token, input)
    hashtag_words = re.findall('#\w+', input)
    for hashtag_word in hashtag_words:
        hashtag_replacement = hashtag_token + " " + hashtag_word[1:]
        input = input.replace(hashtag_word, hashtag_replacement)
    input = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url_token, input)
    # remove numbers and parenthesis
    input = split_sequence(input)
    return input


# category from output
def category_from_output(outputs, all_categories):
    top_n, top_i = outputs.data.topk(1)
    category_i = top_i.view(-1)
    # print all_categories[category_i]
    # print(category_i.size(0))
    categories = [all_categories[category_i[i]] for i in range(category_i.size(0)) ]
    category_ind = [category_i[i] for i in range(category_i.size(0))]
    return categories, category_ind
    # top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    # category_i = top_i[0][0]
    # return all_categories[category_i], category_i