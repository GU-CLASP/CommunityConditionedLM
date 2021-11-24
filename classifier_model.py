import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from IPython import embed

class LSTMClassifier(nn.Module):

    def __init__(self, n_tokens, n_comms, hidden_size, n_layers, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.token_embed = nn.Embedding(n_tokens, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, 1, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1, dropout=dropout)
        self.classifier = nn.Linear(hidden_size*3, n_comms, bias=True)

    def forward(self, text):
        x_e = self.token_embed(text)
        x_e = self.drop(x_e)
        x_1, _  = self.lstm1(x_e)
        x_1 = self.drop(x_1)
        x_2, _  = self.lstm2(x_1)
        x_2 = self.drop(x_2)
        x = torch.cat([x_e, x_1, x_2], dim=-1)
        x = self.classifier(x)
        x, _ = torch.max(x, dim=0)
        return F.log_softmax(x, dim=-1)

    def depth_stratified_preds(self, text):
        """ For prediction only -- predict based on each level of the model."""

        x_e  = self.token_embed(text)
        x_1, _ = self.lstm1(x_e)
        x_2, _ = self.lstm2(x_1)

        x = torch.cat([x_e, x_1, x_2], dim=-1)
        y = self.classifier(x)

        h_size = int(self.classifier.in_features/3)
        y_e = F.linear(x_e, self.classifier.weight[:,:h_size])
        y_1 = F.linear(x_1, self.classifier.weight[:,h_size:2*h_size])
        y_2 = F.linear(x_2, self.classifier.weight[:,2*h_size:])


        p = F.softmax(y, dim=-1)
        p_e = F.softmax(y_e, dim=-1)

        info_gain = divergence(p, p_e)

        return info_gain

        return y_e, y_1, y_2

def divergence(p, q):
    return (p * (p/q).log()).sum(-1)
