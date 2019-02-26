import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
torch.manual_seed(111)
class semiCRF(nn.Module):
    def __init__(self, tagset_size=2, max_seq_len=4):
        '''
        Note in this model, suppose a beginning node
        '''
        super(semiCRF, self).__init__()
        self.tagset_size = tagset_size
        #self.START_TAG = tagset_size - 1
        self.max_seq_len = max_seq_len
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, 
                                                    self.tagset_size))
        
    def _forward_alg(self, span_feats, masks):
        '''
        span_feats: segmentation features, i, j, s
        '''
        #init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        batch_size, sent_len, sent_len, tag_size = span_feats.size()
        forward_var = torch.full((batch_size, self.tagset_size), -10000.).type_as(span_feats)#sent_len*tag_size
        forward_vars = []
        for pos in range(sent_len):
            mask_t = masks[:, pos].unsqueeze(1)#batch_size*1
            if pos == 0:
                #trans = self.transitions[:, self.START_TAG].view(1, -1)
                emit = span_feats[:, 0, 0]#batch_size*tag_size
                forward_var = emit#trans + emit
            else:
                alphas_t = []#record alpha value, alpha(i, s)
                for next_tag in range(self.tagset_size):
                    next_tag_expr_L = []
                    #Remove loops to speed the computation
                    #score = forward_var[:, next_tag].unsqueeze(1)#batch_size*1
                    if pos>=1:
                        l = 0
                        emit_score = span_feats[:, pos-l, pos, next_tag].unsqueeze(1).repeat(1, tag_size)#batch*tag_size
                        #print(pos-l, pos)
                        trans_score = self.transitions[next_tag].repeat(batch_size, 1)#batch，tag_size
                        #print(pos, emit_score.size(), trans_score.size())
                        next_tag_var = forward_vars[pos-l-1] + trans_score + emit_score#batch，tag_size
                        log_sum = log_sum_exp(next_tag_var)#sum of previous alpha and states,#batch
                        next_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                    if pos>=2:
                        l = 1
                        emit_score = span_feats[:, pos-l, pos, next_tag].unsqueeze(1).repeat(1, tag_size)#batch*tag_size
                        #print(pos-l, pos)
                        trans_score = self.transitions[next_tag].repeat(batch_size, 1)#batch，tag_size
                        #score = forward_vars[pos-l-1]
                        next_tag_var = forward_vars[pos-l-1] + trans_score + emit_score#batch，tag_size
                        log_sum = log_sum_exp(next_tag_var)#sum of previous alpha and states
                        next_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                    if pos>=3:
                        l = 2
                        emit_score = span_feats[:, pos-l, pos, next_tag].unsqueeze(1).repeat(1, tag_size)#batch*tag_size
                        #print(pos-l, pos)
                        trans_score = self.transitions[next_tag].repeat(batch_size, 1, 1).squeeze(1)#batch，tag_size
                        next_tag_var = forward_vars[pos-l-1] + trans_score + emit_score#batch，tag_size
                        log_sum = log_sum_exp(next_tag_var)#sum of previous alpha and states
                        
                        next_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                    if pos>=4:
                        l = 3
                        emit_score = span_feats[:, pos-l, pos, next_tag].unsqueeze(1).repeat(1, tag_size)#batch*tag_size
                        #print(pos-l, pos)
                        trans_score = self.transitions[next_tag].repeat(batch_size, 1)#batch，tag_size
                        #score = forward_vars[pos-l-1]
                        next_tag_var = forward_vars[pos-l-1] + trans_score + emit_score#batch，tag_size
                        log_sum = log_sum_exp(next_tag_var)#sum of previous alpha and states,batch*1
                        #score_t = log_sum.unsqueeze(1) * mask_t + score * (1-mask_t)#BATCH_size*1
                        next_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                        
                    next_tag_expr_L = torch.cat(next_tag_expr_L, 1).view(batch_size, -1)#batch*L
                    next_tag_expr = log_sum_exp(next_tag_expr_L).view(batch_size, -1)#batch*1
                    alphas_t.append(next_tag_expr)
                score = torch.cat(alphas_t, 1).view(batch_size, -1)#batch*tag_size
                forward_var = score * mask_t + forward_var * (1-mask_t)
            forward_vars.append(forward_var)
        forward_vars = torch.stack(forward_vars, 1)
        terminal_var = forward_var#.view(batch_size, -1)# + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha, forward_vars
    
    
    def _backward_alg(self, span_feats, masks):

        # Wrap in a variable so that we will get automatic backprop
        batch_size, sent_len, sent_len, tag_size = span_feats.size()
        backward_var = torch.full((batch_size, self.tagset_size), -10000.).type_as(span_feats)#sent_len*tag_size
        backward_vars = [backward_var] * sent_len
        
        span_feats_new = torch.zeros_like(span_feats)
        sent_lens = masks.sum(1).long()
        #move the numbers to the tail
        for i in range(batch_size):
            i_sent_len = sent_lens[[i]]
            for j in range(sent_len):#create a new tensor to record reversed features
                lower = min(sent_len, j+4)
                for k in range(j, lower):
                    span_feats_new[i, j, k] = span_feats[i, i_sent_len-1-k, i_sent_len-1-j]
        
        # Iterate through the sentence
        #feats:n*label_size
        for pos in range(sent_len):
            mask_t = masks[:, pos].unsqueeze(1)#batch_size*1
            if pos == 0:
                backward_var[:, :] = 0.
            else:
                betas_t = []#record alpha value, alpha(i, s)
                for prev_tag in range(self.tagset_size):
                    prev_tag_expr_L = []
                    #Remove loops to speed the computation
                    if pos >= 1:
                        l = 0
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats_new[:, pos-l-1, pos-1]#batch_size*tag_size
                        #print(pos+1, pos+l+1)
                        trans_score = self.transitions[:, prev_tag].repeat(batch_size, 1)#batch，tag_size
                        score = backward_vars[pos-l-1]
                        prev_tag_var = score + trans_score + emit_score
                        log_sum = log_sum_exp(prev_tag_var)#sum of previous alpha and states
                        prev_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                    if pos >=2 :
                        l = 1
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats_new[:, pos-l-1, pos-1]#batch_size*tag_size
                        #print(pos+1, pos+l+1)
                        trans_score = self.transitions[:, prev_tag].repeat(batch_size, 1)#1，tag_size
                        score = backward_vars[pos-l-1]
                        prev_tag_var = score + trans_score + emit_score
                        log_sum = log_sum_exp(prev_tag_var)#sum of previous alpha and states
                     
                        prev_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                    if pos >=3 :
                        l = 2
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats_new[:, pos-l-1, pos-1]#batch_size*tag_size
                        #print(pos+1, pos+l+1)
                        trans_score = self.transitions[:, prev_tag].repeat(batch_size, 1)#batch，tag_size
                        score = backward_vars[pos-l-1]
                        prev_tag_var = score + trans_score + emit_score
                        log_sum = log_sum_exp(prev_tag_var)#sum of previous alpha and states
    
                        prev_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                    if pos >= 4:
                        l = 3
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats_new[:, pos-l-1, pos-1]#batch_size*tag_size
                        #print(pos+1, pos+l+1)
                        trans_score = self.transitions[:, prev_tag].repeat(batch_size, 1)#batch，tag_size
                        score = backward_vars[pos-l-1]#batch*tagsize
                        prev_tag_var = score + trans_score + emit_score
                        log_sum = log_sum_exp(prev_tag_var)#sum of previous alpha and states
                        prev_tag_expr_L.append(log_sum.view(batch_size, -1))#recordd each previous value
                    prev_tag_expr_L = torch.cat(prev_tag_expr_L, 1).view(batch_size,-1)#batch_size*L, stack of each previous node
                    prev_tag_expr = log_sum_exp(prev_tag_expr_L).view(batch_size, -1)#batch_size*1, sum of previous node
                    betas_t.append(prev_tag_expr)
                score_t = torch.cat(betas_t, 1)#batch_size*tag_size
                
                backward_var = score_t * mask_t + backward_var * (1-mask_t)
            backward_vars[pos] = backward_var
        #need to reverse betas
        #trans = self.transitions[:,self.START_TAG].view(1, -1)
        emit = span_feats[:, 0, 0]#batch_size*tag_size
        terminal_var = backward_var + emit#batch_size*tag_size 
        backward_vars = torch.stack(backward_vars, 1) #batch_size*max_len*tag_size 
        backward_vars = backward_vars[:, list(reversed(range(sent_len))), :]
        beta = log_sum_exp(terminal_var)
        return beta, backward_vars
    
    def build_tag_graph(self, feats, masks):
        '''
        Create feature functions
        Args:
        feats: batch_size*max_len*tag_size, a sentence projection
        masks: batch_size*max_len, binary
        '''
        batch_size, max_len, tag_size = feats.size()
        span_scores = torch.zeros(batch_size, max_len, max_len, tag_size).type_as(feats)
        for i in range(max_len):
            upper_bound = min(i+4, max_len)
            for j in range(i, upper_bound):
                mask = masks[:, i:(j+1)]#batch_size*(j+1-i)
                total = mask.sum(1)#batch_size*1
                #total = total.repeat(1, j+1-i)#batch_size*(j+1-i)
                mask = mask.expand(tag_size, batch_size, j+1-i).transpose(0,1).transpose(1,2)
                emit = feats[:, i:(j+1)]*mask#batch_size*(j+1-i)*tag_size
                #print(emit)
                span_score, _ = torch.max(emit, 1)#tag_size*tag_size
                span_scores[:, i, j, :] = span_score#tag_size
        return span_scores
    
    def compute_feat_marginal(self, feats, masks, padding=False):
        batch_size, max_len, max_len, tag_size = feats.size()
        sent_lens = masks.sum(1).long()
        #span_scores = self.build_tag_graph(feats, masks)
        a, f= self._forward_alg(feats, masks)
        z, b =self._backward_alg(feats, masks)
        marginals = []
        marginals_padding = torch.zeros(batch_size, max_len, tag_size).type_as(feats)
        for i in range(batch_size):
            sent_len = sent_lens[i]
            marginal = f[i, :sent_len] + b[i, -sent_len:] - a[i]
            marginals.append(torch.exp(marginal))
            marginals_padding[i, :sent_len] = torch.exp(marginal)
        if padding:
            return marginals_padding
        return marginals
    
    def compute_marginal(self, feats, masks, padding=False):
        batch_size, sent_len, tag_size = feats.size()
        sent_lens = masks.sum(1).long()
        span_scores = self.build_tag_graph(feats, masks)
        a, f= self._forward_alg(span_scores, masks)
        z, b =self._backward_alg(span_scores, masks)
        marginals = []
        marginals_padding = torch.zeros(batch_size, sent_len, tag_size).type_as(feats)
        for i in range(batch_size):
            sent_len = sent_lens[i]
            marginal = f[i, :sent_len] + b[i, -sent_len:] - a[i]
            marginals.append(torch.exp(marginal))
            marginals_padding[i, :sent_len] = torch.exp(marginal)
        if padding:
            return marginals_padding
        return marginals
        
        
    def _viterbi_decode(self, span_feats):
        '''One sentence each time
        Args:
        span_feats: max_len*max_len*tag_size
        sent_len: 
        '''
        sent_len = span_feats.size(0)
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars#1*tag_size
        for pos in range(sent_len):
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            #for each state
            if pos == 0:
                emit = span_feats[0, 0].view(1,-1)
                forward_var = emit#1, tag_size
                best_tag_id = argmax(forward_var)#choose best tag
                best_L = 0#the first one has no steps
                best_tag_id_L = best_tag_id#choose the best L
                next_tag_expr_L = [forward_var[0, best_tag_id]]#record the best value
                for next_tag in range(self.tagset_size):
                    bptrs_t.append((best_L, next_tag))
                #viterbivars_t.append(next_tag_expr_L[best_L].view(1, -1))
            else:
                for next_tag in range(self.tagset_size):
                    best_tag_list = []
                    next_tag_expr_L = []
                    for l in np.arange(self.max_seq_len):
                        if pos < l+1:
                            continue
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats[pos-l, pos][next_tag].view(
                        1, -1).expand(1, self.tagset_size)#1*tag_size
                        trans_score = self.transitions[next_tag].view(1, -1)#1，tag_size
                        next_tag_var = forward_var.view(1, -1) + trans_score + emit_score#1*tag_size
                        best_tag_id = argmax(next_tag_var)#choose best tag
                        best_tag_list.append(best_tag_id)#record best tag for each step L
                        #record each best value
                        next_tag_expr_L.append(next_tag_var[0, best_tag_id].view(1, -1))#recordd each previous value
                
                    best_L = argmax(torch.cat(next_tag_expr_L).view(1, -1))#best L for current point
                    #best_tag_list, a list of L
                    best_tag_id_L = best_tag_list[best_L]#best tag id
                    bptrs_t.append((best_L, best_tag_id_L))#record the best step and tag
                    viterbivars_t.append(next_tag_expr_L[best_L].view(1, -1))#1*1, record  the best value
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
                forward_var = torch.cat(viterbivars_t, 0).view(1, -1)#1*tag_size
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var# + self.transitions[self.tag_to_ix[STOP_TAG]]#1*tag_size
        best_tag_id = argmax(terminal_var)
        #print(log_sum_exp(terminal_var))
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        pos = sent_len - 1
        best_path = []#[(pos, pos, best_tag_id)]
        #backpointers = reversed(backpointers)
        bptrs_t = backpointers[pos]
        while pos >= 0:
            print(pos)
            current_best_tg_id = best_tag_id
            best_L, best_tag_id = bptrs_t[best_tag_id]
            best_path.append((pos-best_L, pos, current_best_tg_id))
            pos = pos-best_L -1
            bptrs_t = backpointers[pos]

        best_path.reverse()
        return path_score, best_path
                    
                
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

    
def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))               