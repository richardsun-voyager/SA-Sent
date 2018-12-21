import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import utils as nn_utils
import torch.nn.init as init
def init_ortho(module):
    for weight_ in module.parameters():
        if len(weight_.size()) == 2:
            init.orthogonal_(weight_)

class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.Wr.state_dict()['weight'])
        self.Ur = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.Ur.state_dict()['weight'])
        self.W = nn.Linear(input_size, hidden_size)
        init.xavier_normal(self.W.state_dict()['weight'])
        self.U = nn.Linear(hidden_size, hidden_size)
        init.xavier_normal(self.U.state_dict()['weight'])

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = F.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = F.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h

class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = Variable(torch.zeros(self.hidden_size)).cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C

class DNM(nn.Module):
    def __init__(self, config, pass_num=5):
        super(DNM, self).__init__()
        self.e = None#eposodic vector
        self.m = None#memory
        
        hidden_dim = config.l_hidden_size
        self.config = config
        self.hidden_dim = hidden_dim
        self.pass_num = pass_num

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.gate_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.memory_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gate_c = nn.Linear(hidden_dim, hidden_dim)
        self.gate_layer1 = nn.Linear(7*hidden_dim, hidden_dim)
        self.gate_layer2 = nn.Linear(hidden_dim, 1)
        init_ortho(self.gate_gru)
        init_ortho(self.memory_gru)

    def forward(self, fact_hiddens, q, fact_lengths):
        '''
        fact_hiddens:batch_size*fact_num*hidden_size
        q:batch_size*hidden_size
        fact_lengths: batch_size
        '''
        m = q
        for i in range(self.pass_num):
            e = self.encoder(fact_hiddens, m, q, fact_lengths)
            m = self.update_memory(e, m)
        return m



    def encoder(self, fact_hiddens, m, q, fact_lengths):
        '''
        Multi-hop episodic memory update
        Args:
        fact_hiddens: batch_size*fact_num*hidden_size
        m: memory, batch_size*hidden_size
        fact_lengths: fact length, batch_size
        '''
        batch_num, sen_num, hidden_size = fact_hiddens.size()
        q = q.expand(sen_num, batch_num, hidden_size).transpose(0, 1)#batch_size*fact_num*hidden_size
        m = m.expand(sen_num, batch_num, hidden_size).transpose(0, 1)#batch_size*fact_num*hidden_size
        f = fact_hiddens
        
        z = [f, m, q, f*q, f*m, (f-q).abs(), (f-m).abs()]#, prod_cq, prod_cm]#7*hidden_size+2
        # prod_cq = torch.matmul(c, self.gate_c(q.transpose(0, 1)))#1*1
        # prod_cm = torch.matmul(c, self.gate_c(m.transpose(0, 1)))
        #Batch_size,fact_num, 7*hidden_size
        z = torch.cat(z, 2)
        #Batch_size,fact_num, hidden_size
        g = self.tanh(self.gate_layer1(z))
        #Batch_size,fact_num, 1
        g = self.gate_layer2(g)
        g = g.squeeze(2)
        #Batch_size,fact_num
        g = self.softmax(g)
          
        #The number of facts are different for each batch
        e = torch.zeros(batch_num, self.hidden_dim)
        if self.config.if_gpu: e = e.cuda()
        for i, fact_num in enumerate(fact_lengths):
            h = torch.zeros(1, 1, self.hidden_dim)
            if self.config.if_gpu: h = h.cuda()
            for j, c in enumerate(fact_hiddens[i, :fact_num]):
                #gated value
                c = c.unsqueeze(0)#1*hidden_size
                _, hidden = self.gate_gru(c.unsqueeze(0), h)
                h = g[i, j] * hidden[0] + (1-g[i,j]) * h
            e[i] = h[0][0]
        return e

    def update_memory(self, e, m):
        '''
        Update episodic memory
        e: batch_size*hidden_size
        m:batch_size*hidden_size
        '''
        _, m = self.memory_gru(e.unsqueeze(1), m.unsqueeze(0))
        return m[0]

    
    def decoder(self):
        pass