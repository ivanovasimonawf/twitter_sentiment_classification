import re
import torch
import random
import numpy as np
from random import shuffle
from torch.autograd import Variable
from read_data import train_data_lines, test_data_lines, separate_data_labels
from utils.string_utils import split_sequence

use_cuda = torch.cuda.is_available()
seed = 1


# return generator generating 1 batch og batch_size at a time
def generate_batches(data, batch_size, max_length, glove_vector_size, twitter_corpus, all_categories):
    shuffle(data)
    for i in range(0, len(data), batch_size):
        batch_data, batch_labels = separate_data_labels(data[i:i + batch_size])
        if len(batch_labels) < batch_size:
            continue
        batch_tensor, labels_tensor, sizes, lens = pack_batch(batch_data, batch_labels, batch_size, glove_vector_size,
                                                        max_length, twitter_corpus, all_categories)
        yield [batch_tensor, labels_tensor, sizes, lens]


# pad sentences with zero vectors for words until they reach max_length
def pad_sequence(sequence, glove_vector_size, max_length, twitter_corpus):
    padded_sequence = np.zeros((max_length, glove_vector_size))
    all_words_for_vecs = list(filter(None, split_sequence(sequence).split(' ')))
    length = len(all_words_for_vecs)
    index = 0
    for ind in range(len(all_words_for_vecs)):
        if index == max_length:
            length = max_length
            break
        try:
            padded_sequence[index, :] = twitter_corpus.word2vec[all_words_for_vecs[ind]]
            index += 1
        except:
            length -= 1
            continue
    return padded_sequence, length


def sort_sequences(sequences, labels, padded_sequence, sizes):
    zipped = zip(sequences, labels, padded_sequence, sizes)
    zipped.sort(key=lambda item: item[3], reverse=True)
    sorted_sequences = []
    sorted_labels = []
    sorted_padded = []
    sorted_sizes = []
    for z in zipped:
        sorted_sequences.append(z[0])
        sorted_labels.append(z[1])
        sorted_padded.append(z[2])
        sorted_sizes.append(z[3])
    return sorted_sequences, sorted_labels, sorted_padded, sorted_sizes


# pack a batch in a PackedSequence with batch_size
def pack_batch(sequences, labels, batch_size, glove_vector_size, max_length, twitter_corpus, all_categories):
    batch_tensor = torch.zeros((batch_size, max_length, glove_vector_size))
    labels_tensor = torch.zeros((batch_size, 1))
    sizes = []
    padded_seq_list = []
    for seq_ind in range(len(sequences)):
        padded_sequence, length = pad_sequence(sequences[seq_ind], glove_vector_size, max_length, twitter_corpus)
        padded_seq_list.append(padded_sequence)
        sizes.append(length)
    sequences, labels, padded_seq_list, sizes = sort_sequences(sequences, labels, padded_seq_list, sizes)
    for ind in range(len(sequences)):
        batch_tensor[ind, :, :] = torch.LongTensor(padded_seq_list[ind])
        labels_tensor[ind, :] = torch.LongTensor([all_categories.index(labels[ind])])

    labels_ind_for_batch = [all_categories.index(labels[i]) for i in range(len(labels))]
    if use_cuda:
        batch_variable = Variable(batch_tensor.cuda())
        labels_variable = Variable(labels_tensor.cuda())
    else:
        batch_variable = Variable(batch_tensor)
        labels_variable = Variable(labels_tensor)
    return torch.nn.utils.rnn.pack_padded_sequence(batch_variable, sizes,
                                                   batch_first=True), labels_variable, sizes, labels_ind_for_batch


