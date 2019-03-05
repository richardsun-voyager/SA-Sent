import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils
from DynamicMem import DNM
from Layer import SimpleCat
import torch.nn.init as init

def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)
            
class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.config = config

        self.rnn = nn.GRU(config.embed_dim + config.mask_dim, config.l_hidden_size, batch_first=True, num_layers = 2,
            bidirectional=False, dropout=config.l_dropout)
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
        #batch_size*emb_dim
        return h[0]
            
class biLSTM(nn.Module):
    def __init__(self, config):
        super(biLSTM, self).__init__()
        self.config = config

        self.rnn = nn.GRU(config.embed_dim + config.mask_dim, int(config.l_hidden_size / 2), batch_first=True, num_layers = int(config.l_num_layers / 2),
            bidirectional=True, dropout=config.l_dropout)
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
    
class DynamicMemTSA(nn.Module):
    def __init__(self, config):
        '''
        LSTM+Aspect
        '''
        super(DynamicMemTSA, self).__init__()
        self.config = config

        self.bilstm = biLSTM(config)
        self.dnm = DNM(config)
        self.linear = nn.Linear(int(config.l_hidden_size/2), 3)
        
        self.conv = nn.Conv1d(config.l_hidden_size, int(config.l_hidden_size/2), 3, padding=1)


        self.loss = nn.NLLLoss()
        self.tanh = nn.Tanh()
        self.cat_layer = SimpleCat(config)
        self.cat_layer.load_vector()
        init.xavier_normal(self.linear.state_dict()['weight'])
        #Modified by Richard Sun
        
    def compute_scores(self, sents, masks, lens):
        '''
        Args:
        sents: batch_size*max_len*word_dim
        masks: batch_size*max_len
        lens: batch_size
        '''
        
        #Context embeddings
        context = self.bilstm(sents, lens)#Batch_size*sent_len*hidden_dim
        
        context = F.relu(self.conv(context.transpose(1, 2)))
        context = context.transpose(1, 2)
        
        batch_size, max_len, hidden_dim = context.size()
        #Target embeddings
        #Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)

        #Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(hidden_dim, batch_size, max_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context

        
        target_emb_avg = torch.sum(target_emb, 1)/torch.sum(masks, 1)#Batch_size*embedding
        #Expand dimension for concatenation
#         target_emb_avg_exp = target_emb_avg.expand(max_len, batch_size, hidden_dim)
#         target_emb_avg_exp = target_emb_avg_exp.transpose(0, 1)#Batch_size*max_len*embedding


        q = target_emb_avg
        facts = context
        m = self.dnm(facts, q, lens)
        #batch_size*3
        m = F.dropout(m, 0.2, self.training)
        outputs = self.linear(m)
        scores = F.log_softmax(outputs, dim=1)
        return scores

    
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
        scores = self.compute_scores(sents, masks, lens)
        loss = self.loss(scores, labels)
        return loss 

    def predict(self, sents, masks, sent_lens):
        if self.config.if_reset:  self.cat_layer.reset_binary()
        sents = self.cat_layer(sents, masks)
        scores = self.compute_scores(sents, masks, sent_lens)
        _, pred_label = scores.max(1)    
        
        #Modified by Richard Sun
        return pred_label
    
    def get_target_emb(self, masks, context):
        '''
        Get the embeddings of targets
        '''
        batch_size, sent_len, dim = context.size()
        #Find target indices, a list of indices
        target_indices, target_max_len = convert_mask_index(masks)
        target_lens = [len(index) for index in target_indices]
        target_lens = torch.LongTensor(target_lens)

        #Find the target context embeddings, batch_size*max_len*hidden_size
        masks = masks.type_as(context)
        masks = masks.expand(dim, batch_size, sent_len).transpose(0, 1).transpose(1, 2)
        target_emb = masks * context
        #Get embeddings for each target
        if target_max_len<3:
            target_max_len = 3
        target_embe_squeeze = torch.zeros(batch_size, target_max_len, dim)
        for i, index in enumerate(target_indices):
            target_embe_squeeze[i][:len(index)] = target_emb[i][index]
        if self.config.if_gpu: 
            target_embe_squeeze = target_embe_squeeze.cuda()
            target_lens = target_lens.cuda()
        return target_embe_squeeze, target_lens
    
    
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
