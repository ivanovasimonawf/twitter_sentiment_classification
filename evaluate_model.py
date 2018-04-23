import torch
from read_data import test_data_lines, twitter_corpus, all_categories, train_data_lines
from utils.string_utils import category_from_output
# from train import batch_size, max_length_of_seq, glove_vector_size, all_categories
from generate_data import generate_batches

model_path = 'models/model2018-04-18 14:50:56.376638.pt'

rnn_model = torch.load(model_path)
batch_size = 16
max_length_of_seq = 24
glove_vector_size = 300


def evaluate(rnn_model, input, lengths, batch_size):
    hidden = rnn_model.init_hidden()
    output, _ = rnn_model.forward(input, hidden, lengths, batch_size)
    return output


data_generator = generate_batches(train_data_lines, batch_size, max_length_of_seq, glove_vector_size, twitter_corpus, all_categories)

correct_predicted = 0
number_of_batches = 0
print("Test data size: ", len(test_data_lines))

for test_batch in data_generator:
    number_of_batches += 1
    batch_tweet_tensor, _, sizes, labels_ind = test_batch[0], test_batch[1], test_batch[2], test_batch[3]
    output = evaluate(rnn_model, batch_tweet_tensor, sizes, batch_size)
    predictions, predictions_ind = category_from_output(output, all_categories)
    for i in range(len(predictions_ind)):
        if predictions_ind[i] == labels_ind[i]:
            correct_predicted += 1
print("Complete number of batches: ", number_of_batches)
print("Percentage: ", float(correct_predicted)/(batch_size*number_of_batches))

