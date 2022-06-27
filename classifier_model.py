import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from IPython import embed

def exp_normalize(x, axis):
    """ https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/ """
    b = x.max(axis=axis)[0]
    expand_dims = [-1] * len(x.shape)
    expand_dims[axis] = x.shape[axis] 
    b = b.unsqueeze(axis).expand(expand_dims)
    y = (x - b).exp()
    return y / y.sum(axis).unsqueeze(axis).expand(expand_dims)

class UnitaryRNN(nn.Module):
    def __init__(self, seq_size):
        super().__init__()
        self.seq_size = seq_size
        # for creating the upper tringulars
        self.ix_mat = torch.zeros(seq_size,seq_size).long()
        for i in range(0,seq_size):
            for j in range(i+1,seq_size):
                self.ix_mat[i,j] = (i* (2*seq_size - i - 3))//2 + j - 1 + 1

    def forward(self, text):

        device = text.device
        seq_len, batch_size, input_size = text.shape
        x = torch.cat([torch.zeros(text.shape[:-1]).to(device).unsqueeze(-1), text], dim=-1)
        tri = torch.index_select(x, -1, self.ix_mat.flatten().to(device)).reshape((*text.shape[:-1],self.seq_size,self.seq_size))
        tri = tri - tri.transpose(-2, -1)
        exp_mat = torch.matrix_exp(tri)

        # h = torch.zeros(seq_len + 1, batch_size, self.seq_size).to(device)
        # h[0,:,0] = 1 # Initial hidden state is [1,0,0,...,0]
        # steps = range(seq_len)
        # for i in steps:
            # pass
            # h[i+1] = torch.einsum('bij,bi->bj', exp_mat[i], h[i])
        # return h[1:,:,]

        h_n = torch.zeros(batch_size, self.seq_size).to(device)
        h_n[:,0] = 1 # Initial hidden state is [1,0,0,...,0]
        h = []
        steps = range(seq_len)
        for i in steps:
            h.append(torch.einsum('bij,bi->bj', exp_mat[i], h_n))
            h_n = h[-1]
        return torch.stack(h)

class SequenceClassifier(nn.Module):
    def __init__(self, n_tokens, n_comms, embedding_size, seq_size, dropout=0.2, seq_encoder='LSTM', agg_seq='final'):
        super().__init__()
        self.agg_seq = agg_seq
        self.seq_encoder = seq_encoder
        self.drop = nn.Dropout(dropout)
        self.token_embed = nn.Embedding(n_tokens, embedding_size)
        if self.seq_encoder == 'LSTM':
            self.encoder = nn.LSTM(embedding_size, seq_size, 1)
        elif self.seq_encoder == 'Unitary':
            self.encoder = UnitaryRNN(seq_size)
        self.classifier = nn.Linear(seq_size, n_comms, bias=True)

    def forward(self, text, text_lens, agg_seq=None):

        agg_seq = self.agg_seq if not agg_seq else agg_seq
        x_embed = self.token_embed(text)

        if self.seq_encoder == 'LSTM':
            x_hidden, _ = self.encoder(self.drop(x_embed))
        else:
            x_hidden = self.encoder(self.drop(x_embed))

        if agg_seq == 'final':
            x_agg = x_hidden[text_lens-1, torch.arange(text.size(1))]
        elif agg_seq == 'max_pool':
            x_agg = torch.max(x_hidden, 0)[0]
        elif agg_seq == 'random_token':
            random_tokens = (torch.rand(text_lens.shape).to(text_lens.device)*(text_lens-1).float()).long()
            x_agg = x_hidden[random-tokens, torch.arange(text.size(1))]

        y_hat = self.classifier(x_agg)
        return F.softmax(y_hat, dim=-1)


class NaiveBayesUnigram(nn.Module):
    def __init__(self, vocab_size, n_comms, alpha=.01):
        super().__init__()
        self.unigram_freq = nn.Parameter(torch.zeros(n_comms, vocab_size), requires_grad=False)
        self.comm_N = nn.Parameter(torch.zeros(n_comms), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.vocab_size = vocab_size
        self.n_comms = n_comms

    def message_cross_entropy_laplace_incremental(self, m, m_lens, comms, alpha=None):
        alpha = alpha if alpha is not None else self.alpha
        N = self.comm_N[comms]
        denom = N.float() + (self.vocab_size * alpha)
        p = self.unigram_freq[comms].gather(-1, m.T).T  * N.float() # select from frequencies according to community and message contens 
        p[p == 0.0] = alpha # smoothing
        pad_mask = (torch.arange(m.shape[0]).unsqueeze(1).expand(-1, m.shape[1]).to(m.device) < m_lens) # zero out pads 
        return -((p / denom).log()).cumsum(dim=0) * pad_mask.float() 

    def infer_comm_incremental(self, m, m_lens):
        repeat = lambda x: torch.tensor(x).repeat(m.size(1)).to(m.device)
        nlls = torch.stack([self.message_cross_entropy_laplace_incremental(m, m_lens, repeat(c)) for c in range(self.n_comms)]).permute(1,2,0)
        return exp_normalize(-nlls, axis=-1)

    def infer_comm(self, m, m_lens):
        repeat = lambda x: torch.tensor(x).repeat(m.size(1)).to(m.device)
        nlls = torch.stack([self.message_cross_entropy_laplace(m, m_lens, repeat(c)) for c in range(self.n_comms)]).T
        return exp_normalize(-nlls, axis=-1)

    def message_cross_entropy_laplace(self, m, m_lens, comms, alpha=None):
        alpha = alpha if alpha is not None else self.alpha
        N = self.comm_N[comms]
        denom = N.float() + (self.vocab_size * alpha)
        p = self.unigram_freq[comms].gather(-1, m.T).T  * N.float() # select from frequencies according to community and message contens 
        p[p == 0.0] = alpha # smoothing
        pad_mask = (torch.arange(m.shape[0]).unsqueeze(1).expand(-1, m.shape[1]).to(m.device) < m_lens) # zero out pads
        return -((p / denom).log() * pad_mask.float()).sum(axis=0) #/ m_lens.float() not really necessary to length normalize

    def test_cross_entropy_laplace(self,data_iter, alpha=None):
        entropies = []
        for batch_no, batch in enumerate(data_iter):
            m, m_lens = batch.text
            batch_entropy = self.message_cross_entropy_laplace(m, m_lens, batch.community, alpha=alpha)
            entropies += batch_entropy.tolist()
        return sum(entropies) / len(entropies)
    def update_alpha(self, alpha):
        state = self.state_dict()
        state['alpha'] = torch.tensor(alpha)
        self.load_state_dict(state)
