import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pickle
from semiCRF import semiCRF
import torch.nn.init as init
import numpy as np
from torch.nn import utils as nn_utils
from util import *
from Layer import SimpleCat, SimpleCatTgtMasked
from multiprocessing import Pool
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)
            
class dynamicLSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(dynamicLSTM, self).__init__()

        self.rnn = nn.LSTM(input_dim , output_dim, batch_first=True, num_layers = 1,
            bidirectional=False)
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
        lstm_out, (h, c) = self.rnn(pack)
        #Unpack the tensor, get the output for varied-size sentences
        #padding with zeros
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # batch * sent_l * 2 * hidden_states 
        return unpacked, h

            
class biLSTM(nn.Module):
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.config = config

        self.rnn = nn.GRU(config.embed_dim+config.mask_dim , int(config.l_hidden_size / 2), batch_first=True, num_layers = 2,
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
class SemiAspectSent(nn.Module):
    def __init__(self, config):
        '''
        LSTM+Aspect
        '''
        super(SemiAspectSent, self).__init__()
        self.config = config

        input_dim = config.l_hidden_size
        kernel_num = config.l_hidden_size
        self.conv = nn.Conv1d(input_dim, kernel_num, 3, padding=1)
        
        self.bilstm = biLSTM(config)
        self.lstm = dynamicLSTM(kernel_num, int(kernel_num/2))
        self.feat2tri = nn.Linear(int(kernel_num/2), 2)
        self.inter_crf = semiCRF()
        self.feat2label = nn.Linear(kernel_num, 3)

        self.loss = nn.NLLLoss()
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(config.dropout2)
        #Modified by Richard Sun
        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()


    def build_tag_graph(self, feats, masks):
        '''
        Create feature functions
        Args:
        feats: batch_size*max_len*hidden_dim, a sentence projection
        masks: batch_size*max_len, binary
        '''
        batch_size, max_len, hidden_size = feats.size()
        span_scores = torch.zeros(batch_size, max_len, max_len, 2).type_as(feats)
        for i in range(max_len):
            upper_bound = min(i+4, max_len)
            for j in range(i, upper_bound):
                #span_score = torch.zeros(batch_size, 4*hidden_size).type_as(feats)
                mask = masks[:, i:(j+1)]#batch_size*(j+1-i)
                total = mask.sum(1)##batch_size, sum of non-zeros
                row_mask = (total > 0)#find which rows have data
                total = torch.where(total>0, total, torch.ones(batch_size))#pad one for the zero element
                #get the feature segment concatenation
                _, h = self.lstm(feats[:, i:(j+1)], total.long())
                #span_score[:, :((j+1-i)*hidden_size)] = (feats[:, i:(j+1)]*mask).view(batch_size, -1)
                #make the empty rows zero
                emit = self.feat2tri(h[0])#batch_size, 2
                emit = emit * row_mask.expand(2, batch_size).transpose(0,1).float()
                span_scores[:, i, j, :] = emit#tag_size
        return span_scores
    
    def compute_scores(self, sents, masks, lens, is_training=True):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''

        context = self.bilstm(sents, lens)#Batch_size*sent_len*hidden_dim
        #pos_weights = self.get_pos_weight(masks, lens)#Batch_size*sent_len

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
#         #Expand dimension for concatenation
#         target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, hidden_dim)
#         target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)#Batch_size*max_len*embedding

#         ###Addition model
#         context = context + target_emb_avg_exp#Batch_size*max_len*2embedding
#         #concatenation model
#         context1 = torch.cat([context, target_emb_avg_exp], 2)
        
        ###neural features
        #tri_scores = self.feat2tri(context) #Batch_size*sent_len*2
        
        #Referred to 
        #tri_scores[:, 0] = 0
        word_mask = torch.full((batch_size, max_len), 0).type_as(context)
        for i in range(batch_size):
            word_mask[i, :lens[i]] = 1.0
        feats = self.build_tag_graph(context, word_mask)

        
        marginals = []
        select_polarities = []
        label_scores = []
        best_latent_seqs = []

        marginals = self.inter_crf.compute_feat_marginal(feats, word_mask, True)
        #marginals = self.inter_crf.compute_marginal(tri_scores, masks.type_as(tri_scores), True)
        select_polarities = marginals[:, :, 1]#batch_size, sent_le
        gammas = select_polarities.sum(1)#batch_size
        sent_vs = torch.bmm(select_polarities.unsqueeze(1), context).squeeze(1)#batch_size, hidden_dim
        #sent_vs = [torch.mm(sp.unsqueeze(0), context[i, :(lens[i]-1), :]) for i, sp in enumerate(select_polarities)]
        sent_vs = sent_vs/gammas.repeat(hidden_dim, 1).transpose(0, 1)#normalization
        sent_vs = self.dropout(sent_vs)
        label_scores = self.feat2label(sent_vs)#
        best_latent_seqs = []
        
        #Take the position into consideration
#         pos_weights = pos_weights.expand(hidden_dim, 
#                                          batch_size, max_len).transpose(0, 1).transpose(1, 2)
#         pos_weights = pos_weights.type_as(context)
#         context = context * pos_weights
        
        
#         #Sentences have different lengths, so deal with them one by one
#         for i, tri_score in enumerate(tri_scores):
#             sent_len = lens[i].cpu().item()
#             if sent_len > 1:
#                 tri_score = tri_score[:sent_len, :]#sent_len, 2
#             else:
#                 print('Too short sentence')
#             marginal = self.inter_crf.compute_marginal(tri_score)#sent_len, latent_label_size
#             #Get only the positive latent factor
#             select_polarity = marginal[:, 1]#sent_len, select only positive ones
#             best_latent_seq = self.inter_crf.predict(tri_score)#sent_len
#             marginal = marginal.transpose(0, 1)  # 2 * sent_len
#             sent_v = torch.mm(select_polarity.unsqueeze(0), context[i, :sent_len, :]) # 1*sen_len, 
            
#             #################Refered to Yoon Kim###########
#             #normalization mentioned in structured attention paper 
#             gamma = select_polarity.sum()/2
#             gamma = torch.clamp(gamma, 0.0001, 10e4)
#             sent_v = sent_v/gamma
            
#             if self.training:
#                 sent_v = self.dropout(sent_v)
            
#             label_score = self.feat2label(sent_v).squeeze(0)#label_size
            
#             label_scores.append(label_score)
#             select_polarities.append(select_polarity)
#             best_latent_seqs.append(best_latent_seq)
#             marginals.append(marginal)
        
#         label_scores = torch.stack(label_scores)
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
        sents = self.cat_layer(sents, masks)
        scores, s_prob  = self.compute_scores(sents, masks, lens)
        s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

        pena = F.relu( self.inter_crf.transitions[1,0] - self.inter_crf.transitions[0,0]) + \
            F.relu(self.inter_crf.transitions[0,1] - self.inter_crf.transitions[1,1])
        norm_pen = self.config.C1 * pena + self.config.C2 * s_prob_norm 
        
        #print('Transition Penalty:', pena)
        #print('Marginal Penalty:', s_prob_norm)
        

        scores = F.log_softmax(scores, 1)#Batch_size*label_size
        
        cls_loss = self.loss(scores, labels)


        return cls_loss, norm_pen 

    def predict(self, sents, masks, sent_lens):
        if self.config.if_reset:  self.cat_layer.reset_binary()
        sents = self.cat_layer(sents, masks)
        scores, best_seqs = self.compute_scores(sents, masks, sent_lens, False)
        _, pred_label = scores.max(1)    
        
        #Modified by Richard Sun
        return pred_label, best_seqs
    
# class TgtMskSASent(nn.Module):
#     def __init__(self, config):
#         '''
#         LSTM+Aspect
#         '''
#         super(TgtMskSASent, self).__init__()
#         self.config = config

#         self.bilstm = biLSTM(config)
#         self.feat2tri = nn.Linear(config.l_hidden_size, 2)
#         self.inter_crf = LinearCRF(config)
#         self.feat2label = nn.Linear(config.l_hidden_size, 3)

#         self.loss = nn.NLLLoss()
        
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(0.3)
#         #Modified by Richard Sun
#         self.cat_layer = SimpleCatTgtMasked(config)
#         self.cat_layer.load_vector()
        
#         self.pool = Pool(processes=5) 

    
#     def compute_scores(self, sents, masks, lens):
#         '''
#         Args:
#         sents: batch_size*max_len*word_dim
#         masks: batch_size*max_len
#         lens: batch_size
#         '''
            
#         batch_size, max_len, _ = sents.size()

#         context = self.bilstm(sents, lens)#Batch_size*sent_len*hidden_dim
        
#         tri_scores = self.feat2tri(context) #Batch_size*sent_len*2
        
#         #Take target embedding into consideration
        
        
        
#         marginals = []
#         select_polarities = []
#         label_scores = []
        
#         #params = zip(tri_scores, context, lens)
#         #results = self.pool.map(self.crf_score, params)
#         #Sentences have different lengths, so deal with them one by one
#         for i, tri_score in enumerate(tri_scores):
#             sent_len = lens[i].cpu().item()
#             if sent_len > 1:
#                 tri_score = tri_score[:sent_len, :]#sent_len, 2
#             else:
#                 print('Too short sentence')
#             marginal = self.inter_crf(tri_score)#sent_len, latent_label_size
#             #Get only the positive latent factor
#             select_polarity = marginal[:, 1]#sent_len, select only positive ones

#             marginal = marginal.transpose(0, 1)  # 2 * sent_len
#             sent_v = torch.mm(select_polarity.unsqueeze(0), context[i, :sent_len, :]) # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
#             label_score = self.feat2label(sent_v).squeeze(0)#label_size
#             label_scores.append(label_score)
#             select_polarities.append(select_polarity)
#             #marginals.append(marginal)
        
#         #label_scores, select_polarities = zip(*result)
#         label_scores = torch.stack(label_scores)

#         return label_scores, select_polarities
    
#     def crf_score(self, params):
#         tri_score, context, sent_len = params
#         if sent_len > 1:
#             tri_score = tri_score[:sent_len, :]#sent_len, 2
#         else:
#             print('Too short sentence')
#         marginal = self.inter_crf(tri_score)#sent_len, latent_label_size
#         #Get only the positive latent factor
#         select_polarity = marginal[:, 1]#sent_len, select only positive ones

#         marginal = marginal.transpose(0, 1)  # 2 * sent_len
#         sent_v = torch.mm(select_polarity.unsqueeze(0), context[:sent_len, :]) # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
#         label_score = self.feat2label(sent_v).squeeze(0)#label_size
#         return label_score, select_polarity
            
         

#     def compute_predict_scores(self, sents, masks, lens):
#         '''
#         Args:
#         sents: batch_size*max_len*word_dim
#         masks: batch_size*max_len
#         lens: batch_size
#         '''

#         batch_size, max_len, _ = sents.size()
#         #batch_size*target_len*emb_dim
#         context = self.bilstm(sents, lens)#Batch_size*max_len*hidden_dim
#         tri_scores = self.feat2tri(context) #Batch_size*sent_len*2
        
#         marginals = []
#         select_polarities = []
#         label_scores = []
#         best_latent_seqs = []
#         #Sentences have different lengths, so deal with them one by one
#         for i, tri_score in enumerate(tri_scores):
#             sent_len = lens[i].cpu().item()
#             if sent_len > 1:
#                 tri_score = tri_score[:sent_len, :]#sent_len, 2
#             else:
#                 print('Too short sentence')
#             marginal = self.inter_crf(tri_score)#sent_len, latent_label_size
#             best_latent_seq = self.inter_crf.predict(tri_score)#sent_len
#             #Get only the positive latent factor
#             select_polarity = marginal[:, 1]#sent_len, select only positive ones

#             marginal = marginal.transpose(0, 1)  # 2 * sent_len
#             sent_v = torch.mm(select_polarity.unsqueeze(0), context[i][:sent_len]) # 1*sen_len, sen_len*hidden_dim=1*hidden_dim
#             label_score = self.feat2label(sent_v).squeeze(0)#label_size
#             label_scores.append(label_score)
#             best_latent_seqs.append(best_latent_seq)
        
#         label_scores = torch.stack(label_scores)

#         return label_scores, best_latent_seqs

    
#     def forward(self, sents, masks, labels, lens):
#         '''
#         inputs are list of list for the convenince of top CRF
#         Args:
#         sent: a list of sentences， batch_size*len*emb_dim
#         mask: a list of mask for each sentence, batch_size*len
#         label: a list labels
#         '''

#         #scores: batch_size*label_size
#         #s_prob:batch_size*sent_len
#         if self.config.if_reset:  self.cat_layer.reset_binary()
#         sents = self.cat_layer(sents, masks)
#         scores, s_prob  = self.compute_scores(sents, masks, lens)
#         s_prob_norm = torch.stack([s.norm(1) for s in s_prob]).mean()

#         pena = F.relu( self.inter_crf.transitions[1,0] - self.inter_crf.transitions[0,0]) + \
#             F.relu(self.inter_crf.transitions[0,1] - self.inter_crf.transitions[1,1])
#         norm_pen = self.config.C1 * pena + self.config.C2 * s_prob_norm 

#         scores = F.log_softmax(scores, dim=1)#Batch_size*label_size
        
#         cls_loss = self.loss(scores, labels)


#         return cls_loss, norm_pen 

#     def predict(self, sents, masks, sent_lens):
#         if self.config.if_reset:  self.cat_layer.reset_binary()
#         sents = self.cat_layer(sents, masks)
#         scores, best_seqs = self.compute_predict_scores(sents, masks, sent_lens)
#         _, pred_label = scores.max(1)    
        
#         #Modified by Richard Sun
#         return pred_label, best_seqs
    
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
    