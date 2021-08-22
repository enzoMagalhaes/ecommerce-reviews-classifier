import torch
import torch.nn as nn

from sentiment_classifier.config import config

class RNN(nn.Module):
  def __init__(self,n_vocab,embed_dim,hidden_size,lstm_dropout,output_size):
    super().__init__()

    self.V = n_vocab
    self.D = embed_dim
    self.M = hidden_size
    self.K = output_size

    self.embed = nn.Embedding(self.V,self.D)

    self.rnn = nn.LSTM(
        input_size = self.D,
        hidden_size=self.M,
        num_layers =1,
        batch_first=True,
        dropout=lstm_dropout
    )

    self.fc = nn.Linear(self.M,self.K)

  def forward(self,X):
    h0 = torch.zeros(1,X.size(0),self.M).to(config.DEVICE)
    c0 = torch.zeros(1,X.size(0),self.M).to(config.DEVICE)

    out = self.embed(X)
    out,_ = self.rnn(out,(h0,c0))
    out,_ = torch.max(out,1)
    out = self.fc(out)
    return out 