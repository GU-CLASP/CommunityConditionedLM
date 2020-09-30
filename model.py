"""
Pytorch implementation of conditional language models, conditioned on some categorical variable, here
refered to as a "community", but it could be anything.

The Transformer and LSTM encoders are adapted from the Pytorch language model tutorial here:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CommunityConditionedLM(nn.Module):

    def __init__(self, n_tokens, n_comms, hidden_size, comm_emsize, encoder_before=None, encoder_after=None, use_community=True, dropout=0.5):
        super(CommunityConditionedLM, self).__init__()
        self.n_comms = n_comms
        self.drop = nn.Dropout(dropout)
        self.token_embed = nn.Embedding(n_tokens, hidden_size)
        self.decoder = nn.Linear(hidden_size, n_tokens)
        self.encoder_before = encoder_before
        self.encoder_after = encoder_after
        self._tune_comm = False
        if use_community:
            self.comm_inference = nn.Embedding(n_comms, n_comms)
            self.comm_embed = WeightedEmbedding(n_comms, comm_emsize)
            self.comm_linear = nn.Linear(hidden_size + comm_emsize, hidden_size)
        self.use_community = use_community

    def forward(self, text, comm):
        device = text.device
        x = self.drop(self.token_embed(text))
        if self.encoder_before is not None:
            x = self.drop(self.encoder_before(x))
        if self.use_community:
            if self._tune_comm:
                comm = self.comm_inference(comm).softmax(1)
            else:
                comm = F.one_hot(comm, num_classes=self.n_comms).type(torch.FloatTensor).to(device)
            x_comm = self.comm_embed(comm).repeat(text.shape[0],1,1)
            x = torch.cat((x, x_comm), 2)
            x = self.drop(self.comm_linear(x))
        if self.encoder_after is not None:
            x = self.drop(self.encoder_after(x))
        x = self.decoder(x)
        return F.log_softmax(x, dim=-1)

    def tune_comm(self):
        self._tune_comm = True
        params = list(self.named_parameters())
        for n, p in params:
            if n != 'comm_inference.weight':
                p.requires_grad = False
        return self

class WeightedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(WeightedEmbedding, self).__init__()
        self.padding_idx = padding_idx
        self.weight = torch.nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        return input.mm(self.weight)

class LSTMLM(nn.Module):

    def __init__(self, n_tokens, hidden_size, n_layers, dropout=0.5):
        super(LSTMLM, self).__init__()
        dropout = dropout if n_layers > 1 else 0
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)

    def forward(self, x):
        x, hidden = self.lstm(x)
        return x

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerLM(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, n_tokens, n_heads, hidden_size, n_layers, dropout=0.5):
        super(TransformerLM, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, n_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):

        # Create left-to-right language modelling mask 
        device = x.device
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask

        #x = self.encoder(text) * math.sqrt(self.hidden_size) # TODO: is this needed? on both encoders??
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, self.src_mask)
        return x

def load_model(weights_file, architecture, encoder_layers, condition_community, community_layer_no, 
        vocab_size, n_communities, hidden_size, community_emsize, heads):

    layers_before = community_layer_no
    layers_after = encoder_layers - community_layer_no
    if architecture == 'Transformer':
        encoder_model = TransformerLM
        encoder_args = (vocab_size, heads, hidden_size)
    elif architecture == 'LSTM':
        encoder_model = LSTMLM
        encoder_args = (vocab_size, hidden_size)
    encoder_before = encoder_model(*encoder_args, layers_before, 0) if layers_before > 0 else None
    encoder_after  = encoder_model(*encoder_args, layers_after,  0) if layers_after  > 0 else None
    lm = CommunityConditionedLM(vocab_size, n_communities, hidden_size, community_emsize,
            encoder_before, encoder_after, condition_community, 0)

    lm.load_state_dict(torch.load(weights_file))
    return lm
