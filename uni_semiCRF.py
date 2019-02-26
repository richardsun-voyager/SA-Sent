import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
torch.manual_seed(111)
class semiCRF(nn.Module):
    def __init__(self, tagset_size=3, max_seq_len=4):
        '''
        Note in this model, suppose a beginning node
        '''
        super(semiCRF, self).__init__()
        self.tagset_size = tagset_size
        self.START_TAG = tagset_size - 1
        self.max_seq_len = max_seq_len
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, 
                                                    self.tagset_size))
        
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG, :] = -10000
        #self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        
    def _forward_alg(self, span_feats, sent_len):
        '''
        span_feats: segmentation features, i, j, s
        '''
        #init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        forward_var = torch.full((1, self.tagset_size), -10000.).cuda()#sent_len*tag_size
        forward_vars = []
        for pos in np.arange(sent_len):
            if pos == 0:
                trans = self.transitions[:, self.START_TAG].view(1, -1)
                emit = span_feats[(0, 0)].view(1,-1)
                forward_var = trans + emit
            else:
                alphas_t = []#record alpha value, alpha(i, s)
                for next_tag in np.arange(self.tagset_size):
                    next_tag_expr_L = []
                    for l in np.arange(self.max_seq_len):
                        if pos < l+1:
                            continue
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats[(pos-l, pos)][next_tag].view(
                        1, -1).expand(1, self.tagset_size)#1*tag_size
                        #print(pos-l, pos)
                        trans_score = self.transitions[next_tag].view(1, -1)#1，tag_size
                        next_tag_var = forward_vars[pos-l-1] + trans_score + emit_score
                        log_sum = log_sum_exp(next_tag_var)#sum of previous alpha and states
                        next_tag_expr_L.append(log_sum.view(1, -1))#recordd each previous value
                    next_tag_expr_L = torch.cat(next_tag_expr_L).view(1, -1)#1*L
                    next_tag_expr = log_sum_exp(next_tag_expr_L).view(1, -1)#1*1
                    alphas_t.append(next_tag_expr)
                forward_var = torch.cat(alphas_t).view(1, -1)#tag_size
            forward_vars.append(forward_var)
        forward_vars = torch.cat(forward_vars)
        terminal_var = forward_var.view(1, -1)# + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha, forward_vars
                    
    def _backward_alg(self, span_feats, sent_len):

        # Wrap in a variable so that we will get automatic backprop
        backward_var = torch.full((1, self.tagset_size), -10000.).cuda()#sent_len*tag_size
        backward_vars = [backward_var] * sent_len

        # Iterate through the sentence
        #feats:n*label_size
        for pos in reversed(np.arange(sent_len)):
            if pos == sent_len-1:
                backward_var[0, :] = 0.
            else:
                betas_t = []#record alpha value, alpha(i, s)
                for prev_tag in np.arange(self.tagset_size):
                    prev_tag_expr_L = []
                    for l in np.arange(self.max_seq_len):
                        if pos+l+1 > sent_len-1:
                            continue
                            
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats[(pos+1, pos+l+1)].view(
                        1, -1)#1*tag_size
                        #print(pos+1, pos+l+1)
                        trans_score = self.transitions[:, prev_tag].view(1, -1)#1，tag_size
                        prev_tag_var = backward_vars[pos+l+1] + trans_score + emit_score
                        log_sum = log_sum_exp(prev_tag_var)#sum of previous alpha and states
                        prev_tag_expr_L.append(log_sum.view(1, -1))#recordd each previous value
                    prev_tag_expr_L = torch.cat(prev_tag_expr_L).view(1,-1)#1*L, stack of each previous node
                    prev_tag_expr = log_sum_exp(prev_tag_expr_L).view(1, -1)#1*1, sum of previous node
                    betas_t.append(prev_tag_expr)
                backward_var = torch.cat(betas_t).view(1, -1)#tag_size
            backward_vars[pos] = backward_var
        #need to reverse betas
        trans = self.transitions[:,self.START_TAG].view(1, -1)
        emit = span_feats[(0, 0)].view(1,-1)
        terminal_var = backward_var + trans + emit
        backward_vars = torch.cat(backward_vars) 
        beta = log_sum_exp(terminal_var)
        return beta, backward_vars
    
    def _score_sentence(self, span_feats, chunks):
        # Gives the score of a provided tag sequence
        #This called golden value
        #chunks: [start_index, end_index, label]
        #trans = self.transitions[:,self.tag_to_ix[START_TAG]].view(1, -1)
        #emit = span_feats[(0, 0)].view(1,-1)
        ##from the start_tag
        score = torch.zeros(1)
        current_label = self.START_TAG
        for i, chunk in enumerate(chunks):
            next_label = chunk[2]
            start = chunk[0]
            end = chunk[1]
            emit = span_feats[(start, end)][current_label]
            score = score + \
                self.transitions[next_label, current_label] + emit
            current_label = next_label#record previous label
        #score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, span_feats, sent_len):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars#1*tag_size
        for pos in np.arange(sent_len):
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            #for each state
            if pos == 0:
                trans = self.transitions[:,self.START_TAG].view(1, -1)
                emit = span_feats[(0, 0)].view(1,-1)
                forward_var = trans + emit
                best_tag_id = argmax(forward_var)#best tag
                best_L = 0
                best_tag_id_L = best_tag_id
                next_tag_expr_L = [forward_var[best_tag_id]]
            else:
                for next_tag in range(self.tagset_size):
                    best_tag_list = []
                    next_tag_expr_L = []
                    for l in np.arange(self.max_seq_len):
                        if pos < l+1:
                            continue
                        #select a span segmentation start for pos-l to pos
                        emit_score = span_feats[(pos-l, pos)][next_tag].view(
                        1, -1).expand(1, self.tagset_size)#1*tag_size

                        trans_score = self.transitions[next_tag].view(1, -1)#1，tag_size
                        next_tag_var = forward_var.view(1, -1) + trans_score + emit_score#1*tag_size
                        best_tag_id = argmax(next_tag_var)#best tag
                        best_tag_list.append(best_tag_id)
                        #record each best value
                        next_tag_expr_L.append(next_tag_var[0, best_tag_id].view(1, -1))#recordd each previous value
                
                best_L = argmax(torch.cat(next_tag_expr_L).view(1, -1))#best L
                best_tag_id_L = best_tag_list[best_L]#best tag id

            #next_tag_var = forward_var + self.transitions[next_tag].view(1, -1)
            #best_tag_id = argmax(next_tag_var)
            bptrs_t.append((best_L, best_tag_id_L))
            viterbivars_t.append(next_tag_expr_L[best_L].view(1, -1))
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
        best_path = [(pos, pos, best_tag_id)]
        #backpointers = reversed(backpointers)
        
        while pos > 0:
            current_best_tg_id = best_tag_id
            bptrs_t = backpointers[pos]
            best_L, best_tag_id = bptrs_t[best_tag_id]
            best_path.append((pos-best_L-1, pos, best_tag_id_L))
            pos = pos-best_L-1
            
        start = best_path.pop()
        #assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    def build_tag_graph(self, feat):
        '''
        Create feature functions
        Args:
        feat: max_len*tag_size, a sentence projection
        '''
        max_len, tag_size = feat.size()
        span_scores = {}
        for i in np.arange(max_len):
            upper_bound = min(i+self.max_seq_len, max_len)
            for j in np.arange(i, upper_bound):
                #traverse tags, 1*tag_size
                #phi = sum(e_i) + b_{i-1}{i}
                emit = (feat[i]+feat[j])/2#tag_size
                span_score = emit#tag_size*tag_size
                span_scores[(i, j)] = span_score#tag_size
        return span_scores
    
    def compute_marginal(self, feat):
        sent_len, tag_size = feat.size()
        span_scores = self.build_tag_graph(feat)
        a, f= self._forward_alg(span_scores, sent_len)
        z, b =self._backward_alg(span_scores, sent_len)
        marginals = f + b - z
        return torch.exp(marginals)
    
    def predict(self, feat):
        return [1,0,1,0,1]
        

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
        