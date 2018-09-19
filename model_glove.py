from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import pickle
from CRF import LinearCRF
import math
from util import *
from torch.nn import utils as nn_utils
from Layer import GloveMaskCat
import torch.nn.init as init
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

class MLSTM(nn.Module):
    def __init__(self, config):
        super(MLSTM, self).__init__()
        self.config = config

        self.rnn = nn.LSTM(300 , int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2),
            bidirectional=True, dropout=config.l_dropout)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        #FIXIT: doesn't have batch
        #Sort the lengths
        # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        # feats = feats[perm_idx]
        #feats = feats.unsqueeze(0)
        pack = nn_utils.rnn.pack_padded_sequence(feats, 
                                                 seq_lengths, batch_first=True)
        
        
        #assert self.batch_size == batch_size
        lstm_out, _ = self.rnn(pack)
        #lstm_out, (hid_states, cell_states) = self.rnn(feats)

        #Unpack the tensor, get the output for varied-size sentences
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        #FIXIT: for batch
        #lstm_out = lstm_out.squeeze(0)
        # batch * sent_l * 2 * hidden_states 
        return unpacked

# consits of three components
class attTSA(nn.Module):
    def __init__(self, config):
        super(attTSA, self).__init__()
        self.config = config
        self.cat_layer = GloveMaskCat(config)

        self.cat_layer.load_vector()

        self.lstm = MLSTM(config)
        self.target2vec = nn.Linear(300, config.l_hidden_size)
        self.vec2label = nn.Linear(config.l_hidden_size, 3)
        self.concatvec_linear = nn.Linear(2*config.l_hidden_size, 1)

        self.cri = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(config.dropout)
        #Modified by Richard Sun
        #If we use Elmo, don't load GloVec
        #self.cat_layer.load_vector()

    def compute_score(self, sent, mask, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sent: a list of sentences， batch_size*len*emb_dim
        mask: a list of mask for each sentence, batch_size*len
        label: a list labels
        '''

        #Get the target embedding
        
        batch_size, sent_len, dim = sent.size()
        #Broadcast mask into3-dimensional
        target_count = mask.sum(1).type_as(sent)#Count the number of target words for each sentence
        #Batch_size * sent_len * emb_dim
        mask = mask.expand(dim, batch_size, sent_len).transpose(0, 1).transpose(1,2)
        mask = mask.type_as(sent)#Longtensor to floattensor
        #make the target as average embedding of words
        #target = torch.mean(sent * mask, 1)#Batch_size*emb_dim
        target_sum = (sent * mask).sum(1)#Batch_size*emb_dim
        target_count = target_count.expand(dim, batch_size).transpose(0,1)
        target = torch.div(target_sum, target_count)#average embedding

        #reduce dimension
        target_vec = self.target2vec(target)#Batch_size*hidden_dim
        #Get the context embeddings for sentences
        #The output for the padding tokens is zeros.
        context = self.lstm(sent, lens)#Batch_size*sent_len*hidden_dim

        ############################################
        ##Multiplication Attention
        #Compute the attention for each token in the sentence. multiplication attention
        target_vec = target_vec.unsqueeze(2)#Batch_size*hidden_dim*1
        scores = torch.bmm(context, target_vec)#Batch_size*sent_len*1

        #############################################
        ###Addition Attention
        #Expand the dimension, batch_size*sent_len*hidden_dim
        # target_vec = target_vec.expand(sent_len, batch_size, self.config.l_hidden_size).transpose(0, 1)
        # #dimension, batch_size*sent_len*(2hidden_dim)
        # context_target_vec = torch.cat([context, target_vec], 2)
        # scores = self.concatvec_linear(context_target_vec)#Batch_size*sent_len*1

        attentions = F.softmax(scores, 1).transpose(1, 2)#Batch_size*1*sent_len
        sents_vec = torch.bmm(attentions, context).squeeze(1)#Batch_size*hidden_dim
        
        #Dropout
        if self.training:
            sents_vec = self.dropout(sents_vec)

        output = self.vec2label(sents_vec)#Bach_size*label_size

        scores = F.log_softmax(output, dim=1)#Batch_size*label_size
        return scores

   
    def forward(self, sent, mask, label, lens):
        #Sent emb_dim + 50
        sent = self.cat_layer(sent, mask)
        
        sent = F.dropout(sent, p=0.2, training=self.training)
        scores = self.compute_score(sent, mask, lens)
        loss = nn.NLLLoss()
        #cls_loss = -1 * torch.log(scores[label])
        cls_loss = loss(scores, label)

        #print('Transition', pena)
        return cls_loss 

    def predict(self, sent, mask, sent_len):
        sent = self.cat_layer(sent, mask)
        scores = self.compute_score(sent, mask, sent_len)
        _, pred_label = scores.max(1)#Find the max label in the 2nd dimension
        
        #Modified by Richard Sun
        return pred_label


# # consits of three components
# class AspectSent(nn.Module):
#     def __init__(self, config):
#         super(AspectSent, self).__init__()
#         self.config = config
#         self.cat_layer = GloveMaskCat(config)

#         self.lstm = MLSTM(config)
#         self.feat2tri = nn.Linear(config.l_hidden_size, 2)
#         self.inter_crf = LinearCRF(config)
#         self.feat2label = nn.Linear(config.l_hidden_size, 3)

#         self.cri = nn.CrossEntropyLoss()
#         #Modified by Richard Sun
#         #If we use Elmo, don't load GloVec
#         #self.cat_layer.load_vector()

#         if not config.if_update_embed:  self.cat_layer.word_embed.weight.requires_grad = False

    
#     def compute_scores(self, sent, mask, lens):
#         if self.config.if_reset:  self.cat_layer.reset_binary()
#         # self.inter_crf.reset_transition()

#         #sent = torch.LongTensor(sent)
#         #sent = sent
#         #Batch_size
#         sent_vec = self.cat_layer(sent, mask)
#         #print('After concatenation:', sent_vec.size())

#         context = self.lstm(sent_vec, lens)#Batch_size*sent_len*hidden_dim
#         #print('After lstm:', context.size())

#         # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
#         feat_context = context  # batch_size*sent_len * hidden_dim
#         tri_scores = self.feat2tri(feat_context) #Batch_size*sent_len*2
#         marginals = self.inter_crf(tri_scores)#Batch_size*sent_len*2
#         #Get only the positive latent factor
#         select_polarity = marginals[:, :, 1]#batch_size*sent_len, select only positive ones

#         marginals = marginals.transpose(1,2)  # batch_size*2 * sent_len
#         sent_v = torch.bmm(select_polarity.unsqueeze(1), context) # batch_size * 1*feat_dim
#         label_scores = self.feat2label(sent_v).squeeze(1)

#         #print('Label Score', label_scores.size())

#         return label_scores, select_polarity, marginals

#     def compute_predict_scores(self, sent, mask, lens):
#         if self.config.if_reset:  self.cat_layer.reset_binary()
#         # self.inter_crf.reset_transition()

#         #sent = torch.LongTensor(sent)
#         #mask = torch.LongTensor(mask)
#         #1*word_len*emb_dim
#         sent_vec = self.cat_layer(sent, mask)

#         context = self.lstm(sent_vec, lens)
#         #Modified by Richard Sun
#         # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
#         feat_context = context  # batch_size*sent_len * hidden_dim
#         tri_scores = self.feat2tri(feat_context) #Batch_size*sent_len*2
#         #print('model: tri_scores', tri_scores.size())
#         marginals = self.inter_crf(tri_scores)#Batch_size*sent_len*2
#         #Get only the positive latent factor
#         select_polarity = marginals[:, :, 1]#batch_size*sent_len, select only positive ones

#         marginals = marginals.transpose(1,2)  # batch_size*2 * sent_len
#         sent_v = torch.bmm(select_polarity.unsqueeze(1), context) # batch_size * 1*feat_dim
#         label_scores = self.feat2label(sent_v).squeeze(1)

#         # # feat_context = torch.cat([context, asp_v], 1) # sent_len * dim_sum
#         # feat_context = context  # sent_len * dim_sum
#         # tri_scores = self.feat2tri(feat_context)
#         # marginals = self.inter_crf(tri_scores)
#         # select_polarity = marginals[:,1]
#         # sent_v = torch.mm(select_polarity.unsqueeze(0), context) # 1 * feat_dim
#         # label_scores = self.feat2label(sent_v).squeeze(0)

#         best_seqs = self.inter_crf.predict(tri_scores)



#         return label_scores, select_polarity, best_seqs

    
#     def forward(self, sent, mask, label, lens):
#         '''
#         inputs are list of list for the convenince of top CRF
#         Args:
#         sent: a list of sentences， batch_size*len*emb_dim
#         mask: a list of mask for each sentence, batch_size*len
#         label: a list labels
#         '''
#         # scores = self.compute_scores(sents, ents, asps, labels)
#         #scores: batch_size*label_size
#         #s_prob:batch_size*sent_len
#         #marginal_prob:batch_size*2 * sent_len
#         scores, s_prob, marginal_prob = self.compute_scores(sent, mask, lens)


#         pena = F.relu( self.inter_crf.transitions[1,0] - self.inter_crf.transitions[0,0]) + \
#             F.relu(self.inter_crf.transitions[0,1] - self.inter_crf.transitions[1,1])
#         norm_pen = ( self.config.C1 * pena + self.config.C2 * s_prob.norm(1) ) / self.config.batch_size

#         scores = F.log_softmax(scores, dim=1)#Batch_size*label_size
#         loss = nn.NLLLoss()
#         #cls_loss = -1 * torch.log(scores[label])
#         cls_loss = loss(scores, label)

#         #print('Transition', pena)

#         print("cls loss {0} with penalty {1}".format(cls_loss.item(), norm_pen.item()))
#         return cls_loss + norm_pen 

#     def predict(self, sent, mask, sent_len):
#         scores, s_probs, best_seqs = self.compute_predict_scores(sent, mask, sent_len)
#         _, pred_label = scores.max(1)    
        
#         #Modified by Richard Sun
#         return pred_label, best_seqs