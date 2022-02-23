import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from IPython import embed

class LSTMClassifier(nn.Module):
    def __init__(self, n_tokens, n_comms, hidden_size, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.token_embed = nn.Embedding(n_tokens, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, 1, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1, dropout=dropout)
        self.classifier = nn.Linear(hidden_size*3, n_comms, bias=True)

    def feature_layers(self, text):
        x_e = self.token_embed(text)
        x_e = self.drop(x_e)
        x_1, _  = self.lstm1(x_e)
        x_1 = self.drop(x_1)
        x_2, _  = self.lstm2(x_1)
        x_2 = self.drop(x_2)
        return x_e, x_1, x_2

    def forward(self, text):
        x = torch.cat(self.feature_layers(text), dim=-1)
        y_hat_seq = self.classifier(x)
        return F.softmax(torch.max(y_hat_seq, dim=0)[0], dim=-1)

    def depth_stratified_activations(self, text):
        """ For prediction only -- class activations for each layer individually."""
        x_e, x_1, x_2 = self.feature_layers(text)
        h_size = int(self.classifier.in_features/3)
        W = self.classifier.weight
        y_e = F.linear(x_e, W[:,:h_size])
        y_1 = F.linear(x_1, W[:,h_size:2*h_size])
        y_2 = F.linear(x_2, W[:,2*h_size:])
        y = y_e + y_1 + y_2 + self.classifier.bias
        return y_e, y_1, y_2, y

