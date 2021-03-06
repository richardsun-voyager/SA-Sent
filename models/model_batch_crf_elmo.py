import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
from BatchLinearChainCRF import LinearChainCrf
import torch.nn.init as init
from Layer import SimpleCat, SimpleCatTgtMasked
import numpy as np
from torch.nn import utils as nn_utils
from util import *

def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

            
class biLSTM(nn.Module):
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.config = config

        self.rnn = nn.GRU(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2),
            bidirectional=True)
        init_ortho(self.rnn)

    # batch_size * sent_l * dim
    def forward(self, feats, seq_lengths=None):
        '''
        Args:
        feats: batch_size, max_len, emb_dim
        seq_lengths: batch_size
        '''
        pack = nn_utils.rnn.pack_padded_sequence(feats, 
                                                 seq_lengths, batch_first=True)
        
        #batch_size*max_len*hidden_dim
        lstm_out, _ = self.rnn(pack)
        #Unpack the tensor, get the output for varied-size sentences
        #padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return unpacked

# consits of three components
class ElmoAspectSent(nn.Module):
    def __init__(self, config):
        '''
        LSTM+Aspect
        '''
        super(ElmoAspectSent, self).__init__()
        self.config = config

        self.bilstm = biLSTM(config)
        #self.map = nn.Linear(config.embed_dim + config.mask_dim, config.l_hidden_size)
        
#         self.feat2tri = nn.Linear(config.l_hidden_size, 2)
#         self.feat2label = nn.Linear(config.l_hidden_size, 3)

        D = config.l_hidden_size
        Co = int(config.l_hidden_size/2)
        self.conv = nn.Conv1d(D, Co, 3, padding=1)
        
        self.feat2tri = nn.Linear(Co, 2+2)
        self.feat2label = nn.Linear(Co, 3)
        
        self.inter_crf = LinearChainCrf(2+2)

        self.loss = nn.NLLLoss()
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.1)
        #Modified by Richard Sun
        #self.cat_layer = SimpleCat(config)
        self.cat_layer = SimpleCatTgtMasked(config)

    
    def compute_scores(self, sents, masks, lens, is_training=True):
        '''
        Args:
        sents: batch_size*max_len*elmo_dim
        masks: batch_size*max_len
        lens: batch_size
        '''
        context = self.bilstm(sents, lens)#batch_size*max_len*dim
        #context = sents
        
        context = F.relu(self.conv(context.transpose(1, 2)))
        context = context.transpose(1, 2)

        batch_size, max_len, hidden_dim = context.size()
        
#         #Target embeddings
#         #Find target indices, a list of indices
#         target_indices, target_max_len = convert_mask_index(masks)

#         #Find the target context embeddings, batch_size*max_len*hidden_size
#         masks = masks.type_as(context)
#         masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
#         target_emb = masks * context

        
#         target_emb_avg = torch.sum(target_emb, 1)/torch.sum(masks, 1)#Batch_size*embedding
#         #target_emb_avg = torch.max(target_emb, 1)[0]#Batch_size*embedding
#         #Expand dimension for concatenation
#         target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, hidden_dim)
#         target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)#Batch_size*max_len*embedding
        
        #context2 = context + target_emb_avg_exp
        
        feats = self.feat2tri(context) #Batch_size*sent_len*2
        
        #Take target embedding into consideration
        
                #word mask
        word_mask = torch.full((batch_size, max_len), 0)
        for i in range(batch_size):
            word_mask[i, :lens[i]] = 1.0

        marginals = self.inter_crf.compute_marginal(feats, word_mask.type_as(feats))
        #print(word_mask.sum(1))
        select_polarities = [marginal[:, 1] for marginal in marginals]
        gammas = [sp.sum()/2 for sp in select_polarities]
        sent_vs = [torch.mm(sp.unsqueeze(0), context[i, :lens[i], :]) for i, sp in enumerate(select_polarities)]
        sent_vs = [sv/gamma for sv, gamma in zip(sent_vs, gammas)]#normalization
        sent_vs = [self.dropout(sent_v) for sent_v in sent_vs]
        label_scores = [self.feat2label(sent_v).squeeze(0) for sent_v in sent_vs]
        best_latent_seqs = self.inter_crf.decode(feats, word_mask.type_as(feats))
        
#         best_latent_seqs = []
        
#         marginals = []
#         select_polarities = []
#         label_scores = []
#         #Sentences have different lengths, so deal with them one by one
#         for i, tri_score in enumerate(tri_scores):
#             sent_len = lens[i].cpu().item()
#             if sent_len > 1:
#                 tri_score = tri_score[:sent_len, :]#sent_len, 2
#             else:
#                 print('Too short sentence')
#             marginal = self.inter_crf(tri_score)#sent_len, latent_label_size
#             #Get only the positive latent factor
#             best_latent_seq = self.inter_crf.predict(tri_score)#sent_len
#             select_polarity = marginal[:, 1]#sent_len, select only positive ones

#             marginal = marginal.transpose(0, 1)  # 2 * sent_len
#             sent_v = torch.mm(select_polarity.unsqueeze(0), context[i, :sent_len, :]) # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
            
#             #################Refered to Yoon Kim###########
#             #normalization mentioned in structured attention paper 
#             gamma = select_polarity.sum()/2
#             if gamma == 0:
#                 gamma = 0.01
#             sent_v = sent_v/gamma
            
#             #is this necessary
#             sent_v = self.dropout(sent_v)
#             ###########
#             label_score = self.feat2label(sent_v).squeeze(0)#label_size
#             label_scores.append(label_score)
#             select_polarities.append(select_polarity)
#             marginals.append(marginal)
#             best_latent_seqs.append(best_latent_seq)
        
        label_scores = torch.stack(label_scores)
        if is_training:
            return label_scores, select_polarities
        else:
            return label_scores, best_latent_seqs


    
    def forward(self, sents, masks, labels, lens):
        '''
        inputs are list of list for the convenince of top CRF
        Args:
        sent: a list of sentences， batch_size*len*emb_dim
        mask: a list of mask for each sentence, batch_size*len
        label: a list labels
        '''

        #scores: batch_size*label_size
        #s_prob:batch_size*sent_len
        if self.config.if_reset:  self.cat_layer.reset_binary()
        sents = self.cat_layer(sents, masks, True)
        scores, s_prob  = self.compute_scores(sents, masks, lens)
        s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

        pena = F.relu( self.inter_crf.transitions[1,0] - self.inter_crf.transitions[0,0]) + \
            F.relu(self.inter_crf.transitions[0,1] - self.inter_crf.transitions[1,1])
        norm_pen = self.config.C1 * pena + self.config.C2 * s_prob_norm 

        scores = F.log_softmax(scores, dim=1)#Batch_size*label_size
        
        cls_loss = self.loss(scores, labels)

        return cls_loss, norm_pen 

    def predict(self, sents, masks, sent_lens):
        if self.config.if_reset:  self.cat_layer.reset_binary()
        sents = self.cat_layer(sents, masks, True)
        scores, best_seqs = self.compute_scores(sents, masks, sent_lens, False)
        _, pred_label = scores.max(1)    
        
        #Modified by Richard Sun
        return pred_label, scores, best_seqs
    
def convert_mask_index(masks):
    '''
    Find the indice of none zeros values in masks, namely the target indice
    '''
    target_indice = []
    max_len = 0
    try:
        for mask in masks:
            indice = torch.nonzero(mask == 1).squeeze(1).cpu().numpy()
            if max_len < len(indice):
                max_len = len(indice)
            target_indice.append(indice)
    except:
        print('Mask Data Error')
        print(mask)
    return target_indice, max_len
    