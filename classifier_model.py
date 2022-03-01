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

class LSTMClassifier(nn.Module):
    def __init__(self, n_tokens, n_comms, hidden_size, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.token_embed = nn.Embedding(n_tokens, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, n_comms, bias=True)

    def forward(self, text, text_lens, agg_seq='final'):
        x_embed = self.token_embed(text)
        x_hidden, _ = self.lstm(self.drop(x_embed))

        # we could consider other sequence aggregators such as 
        # max/average pool, taking a random token (in training), etc.
        if agg_seq == 'final':
            x_agg = x_hidden[text_lens-1, torch.arange(text.size(1))]
        elif agg_seq == 'random_token':
            random_tokens = (torch.rand(text_lens.shape).to(text_lens.device)*(text_lens-1).float()).long()
            x_agg = x_hidden[random-tokens, torch.arange(text.size(1))]
        elif not agg_seq:
            x_agg = x_hidden

        y_hat = self.classifier(self.drop(x_agg))
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
