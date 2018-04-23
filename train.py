import torch
import torch.nn as nn
import time
import datetime
import math
from model import RNN
from read_data import *
from torch import optim
from torch.autograd import Variable
from generate_data import generate_batches
from utils.string_utils import category_from_output

model_path = "models/model"

epochs = 100
n_hidden = 128
output_size = 3
# za kaggle start
# output_size = 2
# za kaggle end
batch_size = 32
dropout_before_softmax = 0.1
stacked_rnn = 2
learning_rate = 0.1
max_length_of_seq = 24
glove_vector_size = 300  # can be 50, 100, 200, 300

criterion = nn.CrossEntropyLoss()
rnn = RNN(glove_vector_size, n_hidden, output_size, dropout_before_softmax, stacked_rnn)

print("Arguments used for this: \n epochs = ", epochs, "\n n_hidden: ", n_hidden, "\n batch size: ", batch_size,
      "\n dropout: ", dropout_before_softmax, "\n learning rate: ", learning_rate, "\n max len seq: ", max_length_of_seq,
      "\n glove vector: ", glove_vector_size, "\n stacked rnn: ", stacked_rnn)


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(output_tensor, input_tensor, seq_sizes):
    hidden = rnn.init_hidden()

    # set all gradients to 0
    rnn.zero_grad()
    # forward one batch and get the last output and hidden state
    output_train, hidden_output = rnn.forward(input_tensor, hidden, seq_sizes, batch_size)

    # get loss of batch (1D)
    loss_train = criterion(output_train.view(output_train.size(0), output_train.size(2)),
                           output_tensor.type(torch.LongTensor).view(-1))
    # add gradients backward
    loss_train.backward()
    # update parameters
    for p in rnn.parameters():
        p.data.sub_(learning_rate, p.grad.data)

    return output_train, loss_train.data[0]


def evaluate(rnn_model, input_tensor, output_tensor, lengths, batch_size):
    hidden = rnn_model.init_hidden()
    output, _ = rnn_model.forward(input_tensor, hidden, lengths, batch_size)

    loss_train = criterion(output.view(output.size(0), output.size(2)),
                           output_tensor.type(torch.LongTensor).view(-1))
    return output, loss_train.data[0]


start = time.time()
epoch_losses_train = []
all_losses_train = []
epoch_losses_test = []
all_losses_test = []

accuracy_test = []
accuracy_train = []

print_every = 10
for epoch in range(epochs):
    epoch_loss_train = 0
    correct_predicted_train = 0
    data_generator_train = generate_batches(train_data_lines, batch_size, max_length_of_seq, glove_vector_size,
                                            twitter_corpus, all_categories)
    # iterate through all batches
    full_items = 0
    for training_batch in data_generator_train:
        full_items += 1
        batch_tweet_tensor, batch_label_tensor, sizes, labels_ind = \
            training_batch[0], training_batch[1], training_batch[2], training_batch[3]
        output, loss = train(batch_label_tensor, batch_tweet_tensor, sizes)
        # count number of correct predicted sentiments
        predictions, predictions_ind = category_from_output(output, all_categories)
        for i in range(len(predictions_ind)):
            if predictions_ind[i] == labels_ind[i]:
                correct_predicted_train += 1
        all_losses_train.append(loss)
        epoch_loss_train += loss
    accuracy_train.append(correct_predicted_train)
    epoch_losses_train.append(epoch_loss_train)

    print('Epoch: ', epoch)
    print("Correct predicted: ", correct_predicted_train)
    print("Loss: ", float(epoch_loss_train))

    epoch_loss_test = 0
    correct_predicted_test = 0
    number_of_batches = 0
    data_generator_test = generate_batches(test_data_lines, batch_size, max_length_of_seq, glove_vector_size,
                                            twitter_corpus, all_categories)
    for test_batch in data_generator_test:
        number_of_batches += 1
        batch_tweet_tensor, batch_label_tensor, sizes, labels_ind = test_batch[0], test_batch[1], test_batch[2], test_batch[3]
        output, loss = evaluate(rnn, batch_tweet_tensor, batch_label_tensor, sizes, batch_size)
        predictions, predictions_ind = category_from_output(output, all_categories)
        for i in range(len(predictions_ind)):
            if predictions_ind[i] == labels_ind[i]:
                correct_predicted_test += 1
        epoch_loss_test += loss
        all_losses_test.append(loss)
    epoch_losses_test.append(epoch_loss_test)
    accuracy_test.append(correct_predicted_test)

    print("Correct: ", correct_predicted_test, "out of ", len(test_data_lines))
    print("Percentage: ", float(correct_predicted_test)/len(test_data_lines))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(epoch_losses_train)
plt.plot(epoch_losses_test)
plt.legend(["train", "test"], loc="upper left")
plt.show()

plt.figure()
plt.plot(accuracy_test)
plt.plot(accuracy_train)
plt.legend(["test", "train"], loc="upper left")
plt.show()
torch.save(rnn, model_path + str(datetime.datetime.now()) + ".pt")
