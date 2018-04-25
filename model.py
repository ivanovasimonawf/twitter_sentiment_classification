import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, stacked_rnn=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.stacked_rnn = stacked_rnn
        self.gru = nn.GRU(input_size, hidden_size, num_layers=stacked_rnn, batch_first=True)
        self.dropout_out = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2) # beshe po prva dimenzija po koja dimenzija sumata da im e 1


    def forward(self, input, hidden, lengths, batch_size):
        output = input
        output, hidden = self.gru(output, hidden)
        r_out, recovered_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # unpack batch

        chosen_values = Variable(torch.zeros((batch_size, 1, self.hidden_size)))
        ind = 0
        for index in lengths:
            chosen_values[ind, 0, :] = r_out[ind, index-1, :]
            ind += 1

        output = self.out(chosen_values)
        output = self.dropout_out(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(self.stacked_rnn, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result