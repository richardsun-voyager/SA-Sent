import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import pdb
import pickle
import numpy as np
import math
from torch.nn import utils as nn_utils
from util import *
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

            
class charLSTM(nn.Module):
    def __init__(self, char_dim, hidden_dim):
        super(charLSTM, self).__init__()

        self.rnn = nn.GRU(char_dim, hidden_dim, batch_first=True, num_layers = 1,
            bidirectional=False)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, seqs, seq_lengths=None):
        '''
        Args:
        seqs: sent_num, word_num, emb_dim
        seq_lengths: sent_num
        '''
        perm_seq_lens, perm_idx = seq_lengths.sort(0, descending=True)
        _, desorted_perm_idx = torch.sort(perm_idx, descending=False)
        perm_seqs = seqs[perm_idx]
        
        pack = nn_utils.rnn.pack_padded_sequence(perm_seqs, 
                                                 perm_seq_lens, batch_first=True)
        
        #batch_size*max_len*hidden_dim
        lstm_out, h = self.rnn(pack)
        #Unpack the tensor, get the output for varied-size sentences
        #padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return h[0][desorted_perm_idx]
    
class sentLSTM(nn.Module):
    def __init__(self, word_dim, hidden_dim):
        super(sentLSTM, self).__init__()

        self.rnn = nn.GRU(word_dim, hidden_dim, batch_first=True, num_layers = 2,
            bidirectional=True)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, seqs, seq_lengths=None):
        '''
        Args:
        seqs: sent_num, word_num, emb_dim
        seq_lengths: sent_num
        '''
        perm_seq_lens, perm_idx = seq_lengths.sort(0, descending=True)
        _, desorted_perm_idx = torch.sort(perm_idx, descending=False)
        perm_seqs = seqs[perm_idx]
        
        pack = nn_utils.rnn.pack_padded_sequence(perm_seqs, 
                                                 perm_seq_lens, batch_first=True)
        
        #batch_size*max_len*hidden_dim
        lstm_out, h = self.rnn(pack)
        #Unpack the tensor, get the output for varied-size sentences
        #padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return unpacked[desorted_perm_idx]
    
    
class SimpleCat(nn.Module):
    def __init__(self, config):
        '''
        Concatenate word embeddings and target embeddings
        '''
        super(SimpleCat, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.word_num, config.embed_dim)
        self.char_embed = nn.Embedding(config.char_num, config.char_dim)
        self.mask_embed = nn.Embedding(2, config.mask_dim)
        
        #positional embeddings
        self.dropout = nn.Dropout(config.dropout)
        self.crnn = charLSTM(config.char_dim, config.char_dim)
        self.wrnn = sentLSTM(config.embed_dim, config.l_hidden_size)

    # input are tensors
    def forward(self, word_ids, target_mask, sent_lens, batch_char_ids_tensor, batch_char_lens_tensor):
        '''
        Args:
        word_ids: tensor, shape(batch_size, max_len, emb_dim)
        target_mask: tensor, shape(batch_size, max_len)
        batch_char_ids_tensor: tensor, shape(batch_size, max_len, max_char_len)
        batch_char_lens_tensor: batch_size, max_len
        sent_lens: batch_size
        '''
        #Modified by Richard Sun
        #Use ELmo embedding, note we need padding
        sent = Variable(word_ids)
        mask = Variable(target_mask)

        #Use GloVe embedding
        sent_vec = self.word_embed(sent)# batch_siz*sent_len * dim
        sent_vec = self.dropout(sent_vec)
        batch_size, max_word_num, hidden_dim = sent_vec.size()
        
        char_vec = self.char_embed(batch_char_ids_tensor)# batch_siz*sent_len * char_len * char_dim
        mask_vec = self.mask_embed(mask) # batch_size*max_len* dim
        char_output = torch.zeros(len(mask), max_word_num, self.config.char_dim).type_as(sent_vec)
        #print(mask_vec.size())
        for i in range(len(mask)):
            sent_len = sent_lens[i]
            char_emb = char_vec[i, :sent_len]
            char_len = batch_char_lens_tensor[i, :sent_len]
            outputs = self.crnn(char_emb, char_len)#sent_len*hidden_dim
            char_output[i, :sent_len] = outputs
            
        
        #Concatenation
        sent_vec = torch.cat([sent_vec, mask_vec], 2)
        char_output = torch.cat([char_output, mask_vec], 2)

        # for test
        return sent_vec, char_output
    def reset_binary(self):
            self.mask_embed.weight.data[0].zero_()
            
    def load_vector(self):
        '''
        Load pre-savedd word embeddings
        '''
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f)
            print("Loaded from {} with shape {}".format(self.config.embed_path, vectors.shape))
            #self.word_embed.weight = nn.Parameter(torch.FloatTensor(vectors))
            self.word_embed.weight.data.copy_(torch.from_numpy(vectors))
            self.word_embed.weight.requires_grad = self.config.if_update_embed
            print('embeddings loaded')