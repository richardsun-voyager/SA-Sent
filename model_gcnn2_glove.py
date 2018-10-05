import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from torch.nn import utils as nn_utils
class CNN_Gate_Aspect_Text(nn.Module):
    def __init__(self, config):
        '''
        In this model, both context words and target words are processed by gated CNN
        '''
        super(CNN_Gate_Aspect_Text, self).__init__()
        self.config = config
        
        #V = config.embed_num
        D = config.embed_dim
        C = 3#config.class_num

        Co = 128#kernel numbers
        Ks = [1, 2, 3]#kernel filter size

        # self.embed = nn.Embedding(V, D)
        # self.embed.weight = nn.Parameter(args.embedding, requires_grad=True)

        # self.aspect_embed = nn.Embedding(A, args.aspect_embed_dim)
        # self.aspect_embed.weight = nn.Parameter(args.aspect_embedding, requires_grad=True)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])
        self.convs3 = nn.ModuleList([nn.Conv1d(D, Co, K, padding=K-2) for K in [3]])

        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm1d(Co)
        self.fc_aspect = nn.Linear(D, Co)

    
    def compute_score(self, sent, target, lens):
        '''
        gated cnn, note even the aspect words are processed by CNNs
        Args:
        sent: a list of sentences， batch_size*max_len*(2*emb_dim)
        target: a list of target embedding for each sentence, batch_size*len*emb_dim
        label: a list labels
        '''

        #Get the target embedding
        batch_size, sent_len, dim = sent.size()
        #Deal with target
        target_conv = [self.relu(conv(target.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in target_conv]
        aspect_v = torch.cat(aa, 1)

        #Mask the padding embeddings
        pack = nn_utils.rnn.pack_padded_sequence(sent, 
                                                 lens, batch_first=True)
        unpacked, _ = nn_utils.rnn.pad_packed_sequence(pack, batch_first=True)
        #Conv input: batch_size * emb_dim * max_len
        #Conv output: batch_size * out_dim * (max_len-k+1)

        x = [self.tanh(conv(unpacked.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [self.relu(conv(unpacked.transpose(1, 2)) + aspect_v.unsqueeze(2)) for conv in self.convs2]
        x = [i*j for i, j in zip(x, y)] #batch_size * out_dim * (max_len-k+1) * len(filters)

        # pooling method
        x0 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,out_dim), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        sents_vec = torch.cat(x0, 1)#N*(3*out_dim)
        #logit = self.fc1(x0)  # (N,C)

        #Dropout
        if self.training:
            sents_vec = F.dropout(sents_vec, self.config.dropout)

        output = self.fc1(sents_vec)#Bach_size*label_size

        scores = F.log_softmax(output, dim=1)#Batch_size*label_size
        return scores


    def forward(self, sent, target, label, lens):
        #Sent emb_dim + 50
        
        sent = F.dropout(sent, p=0.2, training=self.training)
        scores = self.compute_score(sent, target, lens)
        loss = nn.NLLLoss()
        #cls_loss = -1 * torch.log(scores[label])
        cls_loss = loss(scores, label)

        #print('Transition', pena)
        return cls_loss 

    def predict(self, sent, target, sent_len):
        #sent = self.cat_layer(sent, mask)
        scores = self.compute_score(sent, target, sent_len)
        _, pred_label = scores.max(1)#Find the max label in the 2nd dimension
        
        #Modified by Richard Sun
        return pred_label
