import torch
import torch.nn as nn
import torch.nn.functional as F
class SLSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, window_size=1, gpu=False):
        super(SLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.device = torch.device("cuda" if gpu else "cpu")

        concat_num = window_size * 2 + 1
        self.it_ep = nn.Linear(concat_num*hidden_size, hidden_size)
        self.lt_ep = nn.Linear(concat_num*hidden_size, hidden_size)
        self.rt_ep = nn.Linear(concat_num*hidden_size, hidden_size)
        self.ft_ep = nn.Linear(concat_num*hidden_size, hidden_size)
        self.st_ep = nn.Linear(concat_num*hidden_size, hidden_size)
        self.ot_ep = nn.Linear(concat_num*hidden_size, hidden_size)
        self.ut_ep = nn.Linear(concat_num*hidden_size, hidden_size)
        
        self.it_x = nn.Linear(emb_size, hidden_size)
        self.lt_x = nn.Linear(emb_size, hidden_size)
        self.rt_x = nn.Linear(emb_size, hidden_size)
        self.ft_x = nn.Linear(emb_size, hidden_size)
        self.st_x = nn.Linear(emb_size, hidden_size)
        self.ot_x = nn.Linear(emb_size, hidden_size)
        self.ut_x = nn.Linear(emb_size, hidden_size)
        
        self.it_g = nn.Linear(hidden_size, hidden_size)
        self.lt_g = nn.Linear(hidden_size, hidden_size)
        self.rt_g = nn.Linear(hidden_size, hidden_size)
        self.ft_g = nn.Linear(hidden_size, hidden_size)
        self.st_g = nn.Linear(hidden_size, hidden_size)
        self.ot_g = nn.Linear(hidden_size, hidden_size)
        self.ut_g = nn.Linear(hidden_size, hidden_size)
        
        self.fgg = nn.Linear(hidden_size, hidden_size)
        self.fgu = nn.Linear(hidden_size, hidden_size)
        self.fig = nn.Linear(hidden_size, hidden_size)
        self.fiu = nn.Linear(hidden_size, hidden_size)
        self.og = nn.Linear(hidden_size, hidden_size)
        self.ou = nn.Linear(hidden_size, hidden_size)
        
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, sents, sent_lens, time_steps=6):
        '''
        sents: batch_size*max_len * emb_dim
        sent_lens: batch_size
        '''
        batch_size, max_len, emb_dim = sents.size()
        outputs = torch.zeros(batch_size, max_len, self.hidden_size).to(self.device)
        sent_final_states = torch.zeros(batch_size, self.hidden_size).to(self.device)
        for i, sent in enumerate(sents):
            sent_len = sent_lens[i]
            sent = sent[:sent_len]
            #generate intial states
            hs, cs, g = self.initHidden(sent_len)
            for t in range(time_steps):
                hs, cs, g, cg = self.recurrent_state(sent, 
                                                             hs, cs, g)
            outputs[i, :sent_len] = hs[self.window_size:(-self.window_size)]
            sent_final_states[i] = g
            
        return outputs, sent_final_states
        

    def recurrent_state(self, sent, hs, cs, g):
        '''
        Update recurrent state for each step
        '''
        g, cg = self.sent_state(hs, g, cs)
        #padding the left
        hs_new = [torch.zeros(self.hidden_size).to(self.device)] * self.window_size
        cs_new = [torch.zeros(self.hidden_size).to(self.device)] * self.window_size
        for j, w in enumerate(sent):
            ep = hs[j:(j+2*self.window_size+1)]
            ep = ep.view(-1)
            #hidden_size
            it = self.sigmoid(self.it_ep(ep)+self.it_x(w)+self.it_g(g))
            lt = self.sigmoid(self.lt_ep(ep)+self.lt_x(w)+self.lt_g(g))
            rt = self.sigmoid(self.rt_ep(ep)+self.rt_x(w)+self.rt_g(g))
            ft = self.sigmoid(self.ft_ep(ep)+self.ft_x(w)+self.ft_g(g))
            st = self.sigmoid(self.st_ep(ep)+self.st_x(w)+self.st_g(g))
            ot = self.sigmoid(self.ot_ep(ep)+self.ot_x(w)+self.ot_g(g))
            ut = self.tanh(self.ut_ep(ep)+self.ut_x(w)+self.ut_g(g))
            #softmax
            temp = torch.stack([it, lt, rt, ft, st])
            temp = self.softmax(temp)
            it, lt, rt, ft, st = temp[0], temp[1], temp[2], temp[3], temp[4]
            ct = lt*cs[j] + ft*cs[j+1] + rt*cs[j+2] + st*cg + it*ut
            ht = ot*self.tanh(ct)
            hs_new.append(ht)
            cs_new.append(ct)
        #padding the right
        hs_new.extend([torch.zeros(self.hidden_size).to(self.device)] * self.window_size)
        cs_new.extend([torch.zeros(self.hidden_size).to(self.device)] * self.window_size)
        hs_new = torch.stack(hs_new)
        cs_new = torch.stack(cs_new)  
        return hs_new, cs_new, g, cg
    
    def sent_state(self, hs, g_prev, cs):
        '''
        Compute sentence state
        hs: [h0, h1,..., hn], max_len*hidden_dim
        cs: [c0, c1,..., cn], max_len*hidden_dim
        '''
        h_avg = hs.mean(0)
        cg_prev = cs.mean(0)
        fg = self.sigmoid(self.fgg(g_prev)+self.fgu(h_avg))
        fg = F.softmax(fg, 0)
        o = self.sigmoid(self.og(g_prev)+self.ou(h_avg))
        #update the states
        fi = self.sigmoid(self.fig(g_prev).expand(len(hs), 
                                                  self.hidden_size)+self.fiu(hs))
        fi = F.softmax(fi, 1)
        cg = torch.sum(fi * cs, 0) + fg*cg_prev

        
        g = o*self.tanh(cg)
        return g, cg

    def initHidden(self, sent_len):
        h = torch.rand(self.hidden_size).to(self.device)*0.05-0.025
        g_prev = torch.rand(self.hidden_size).to(self.device)*0.05-0.025
        cs = torch.zeros(sent_len+2*self.window_size, self.hidden_size).to(self.device)
        hs = torch.zeros(sent_len+2*self.window_size, self.hidden_size).to(self.device)
        hs[self.window_size:(sent_len+2*self.window_size-1)] = h.expand(sent_len, 
                                                                        self.hidden_size)
        cs[self.window_size:(sent_len+2*self.window_size-1)] = torch.randn(sent_len, 
                                                                        self.hidden_size).to(self.device)
        
        return hs, cs, g_prev