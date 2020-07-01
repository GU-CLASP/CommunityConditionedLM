import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMLM(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ncomms, nhid, layers_before, layers_after, community_emsize, dropout=0.5):
        super(LSTMLM, self).__init__()
        emsize = nhid # set these sequal so we can uniformly handle 
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, emsize)
        self.community_encoder = nn.Embedding(ncomms, community_emsize)
        self.community_linear = nn.Linear(nhid + community_emsize, nhid)
        if layers_before == 0:
            self.lstm1 = None
        else:
            lstm1_dropout = 0 if layers_before == 1 else dropout
            self.lstm1 = nn.LSTM(nhid, nhid, layers_before, dropout=lstm1_dropout)
        if layers_after == 0:
            self.lstm2 = None
        else:
            lstm2_dropout = 0 if layers_after == 1 else dropout
            self.lstm2 = nn.LSTM(nhid, nhid, layers_after, dropout=lstm2_dropout)

        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.layers_before = layers_before
        self.layers_afer = layers_after

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, text, community):
        x = self.drop(self.encoder(text))
        if self.lstm1 is not None:
            x, hidden = self.lstm1(x)
            x = self.drop(x)
        if community is not None: 
            x_comm = self.community_encoder(community).repeat(text.shape[0],1,1)
            x = torch.cat((x, x_comm), 2)
            x = self.community_linear(x)
        if self.lstm2 is not None:
            x, hidden = self.lstm2(x)
        x = self.drop(x)
        x = self.decoder(x)
        x = x.view(-1, self.ntoken)
        return F.log_softmax(x, dim=1)

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

    def __init__(self, ntoken, ncomms, nhead, nhid, layers_before, layers_after, community_emsize, dropout=0.5):
        super(TransformerLM, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        ninp = nhid # Like the LSTM we use the same size hidden/token embedding
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.community_encoder = nn.Embedding(ncomms, community_emsize)
        self.community_linear = nn.Linear(nhid + community_emsize, nhid)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        if layers_before == 0:
            self.transformer_encoder1 = None
        else:
            self.transformer_encoder1 = TransformerEncoder(encoder_layers, layers_before)
        if layers_after == 0:
            self.transformer_encoder2 = None
        else:
            self.transformer_encoder2 = TransformerEncoder(encoder_layers, layers_after)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, text, community, has_mask=True):
        if has_mask:
            device = text.device
            if self.src_mask is None or self.src_mask.size(0) != len(text):
                mask = self._generate_square_subsequent_mask(len(text)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        x = self.encoder(text) * math.sqrt(self.ninp)
        x = self.pos_encoder(x)
        if self.transformer_encoder1 is not None:
            x = self.transformer_encoder1(x, self.src_mask)
        if community is not None: 
            x_comm = self.community_encoder(community).repeat(text.shape[0],1,1)
            x = torch.cat((x, x_comm), 2)
            x = self.community_linear(x)
        if self.transformer_encoder2 is not None:
            x = self.transformer_encoder2(x, self.src_mask)
        x = self.decoder(x)
        return F.log_softmax(x, dim=-1)
