import torch
from read_data import test_data_lines, twitter_corpus, all_categories, train_data_lines
from utils.string_utils import category_from_output, unicode_to_ascii
# from train import batch_size, max_length_of_seq, glove_vector_size, all_categories
from generate_data import generate_batches

model_path = 'models/model2018-04-30 14:25:14.306869.pt'
file_not_correct_predicted = 'data/wrong_predicted.txt'

rnn_model = torch.load(model_path)
batch_size = 128
max_length_of_seq = 17
glove_vector_size = 300


def evaluate(rnn_model, input, lengths, batch_size):
    hidden = rnn_model.init_hidden(batch_size)
    output, _ = rnn_model.forward(input, hidden, lengths, batch_size)
    return output


data_generator = generate_batches(test_data_lines, batch_size, max_length_of_seq, glove_vector_size, twitter_corpus, all_categories)

correct_predicted = 0
number_of_batches = 0
print("Test data size: ", len(test_data_lines))

file = open(file_not_correct_predicted, 'w')
file.write('Correct' + '\t' + 'Predicted' + '\t' + 'Tweet' + '\n')
negative = 0
positive = 0
wrong_predicted = 0
for test_batch in data_generator:
    number_of_batches += 1
    batch_tweet_tensor, _, sizes, labels_ind, sequences, labels = test_batch[0], test_batch[1], test_batch[2], test_batch[3], test_batch[4], test_batch[5]
    output = evaluate(rnn_model, batch_tweet_tensor, sizes, batch_size)
    predictions, predictions_ind = category_from_output(output, all_categories)
    for i in range(len(predictions_ind)):
        if predictions_ind[i] == labels_ind[i]:
            correct_predicted += 1
        else:
            wrong_predicted += 1
            if labels_ind[i] == 0:
                negative += 1
            else:
                positive += 1
            try:
                file.write(str(labels_ind[i]) + '\t' + str(predictions_ind[i]) + '\t' + unicode_to_ascii(sequences[i]) + '\n')
            except UnicodeEncodeError:
                print(sequences[i])

print("Complete number of batches: ", number_of_batches)
print("Percentage: ", float(correct_predicted)/(batch_size*number_of_batches))
print("Negative as positive: ", negative)
print("Positive as negative: ", positive)
print("Wrong predicted: ", wrong_predicted)
print("Full number of sentences:  ", number_of_batches*batch_size)

