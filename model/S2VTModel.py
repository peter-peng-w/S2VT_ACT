import torch
from torch import nn
import torch.nn.functional as F
import random
from torch.autograd import Variable


class S2VTModel(torch.nn.Module):
    def __init__(self, vocab_size=10000, max_len=28, dim_hidden=512, dim_word=512, dim_vid=2048, sos_id=2, eos_id=1,
                 n_layers=1, rnn_cell='gru', rnn_dropout_p=0.2):
        super().__init__()
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        # First layer RNN
        self.rnn1 = self.rnn_cell(
            dim_vid, dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p
        )
        # Second Layer RNN
        self.rnn2 = self.rnn_cell(
            dim_hidden + dim_word, dim_hidden, n_layers,
            batch_first=True, dropout=rnn_dropout_p
        )

        self.dim_vid = dim_vid
        self.dim_output = vocab_size
        self.dim_hidden = dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(self.dim_output, self.dim_word)

        self.out = nn.Linear(self.dim_hidden, self.dim_output)

    def forward(self, vid_feats, target_variable=None,
                mode='train', opt={}):
        batch_size, n_frames, _ = vid_feats.shape
        padding_words = Variable(vid_feats.data.new(batch_size, n_frames, self.dim_word)).zero_()
        padding_frames = Variable(vid_feats.data.new(batch_size, 1, self.dim_vid)).zero_()
        # padding_words = vid_feats.data.new(batch_size, n_frames, self.dim_word).zero_()
        # padding_frames = vid_feats.data.new(batch_size, 1, self.dim_vid).zero_()
        state1 = None
        state2 = None
        # self.rnn1.flatten_parameters()
        # self.rnn2.flatten_parameters()
        # print('vid_feats shape:{}'.format(vid_feats.shape))
        # print('vid_feats dtype:{}'.format(vid_feats.type()))
        # print('padding_words shape:{}'.format(padding_words.shape))
        # print('padding_words dtype:{}'.format(padding_words.type()))
        output1, state1 = self.rnn1(vid_feats, state1)
        input2 = torch.cat((output1, padding_words), dim=2)
        output2, state2 = self.rnn2(input2, state2)

        seq_probs = []
        seq_preds = []
        if mode == 'train':
            for i in range(self.max_length - 1):
                # <eos> doesn't input to the network
                current_words = self.embedding(target_variable[:, i])
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)

        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            current_words = self.embedding(
                Variable(torch.LongTensor([self.sos_id] * batch_size)).to(device))
            for i in range(self.max_length - 1):
                self.rnn1.flatten_parameters()
                self.rnn2.flatten_parameters()
                output1, state1 = self.rnn1(padding_frames, state1)
                input2 = torch.cat(
                    (output1, current_words.unsqueeze(1)), dim=2)
                output2, state2 = self.rnn2(input2, state2)
                logits = self.out(output2.squeeze(1))
                logits = F.log_softmax(logits, dim=1)
                seq_probs.append(logits.unsqueeze(1))
                _, preds = torch.max(logits, 1)
                current_words = self.embedding(preds)
                seq_preds.append(preds.unsqueeze(1))
            seq_probs = torch.cat(seq_probs, 1)
            seq_preds = torch.cat(seq_preds, 1)
        return seq_probs, seq_preds
