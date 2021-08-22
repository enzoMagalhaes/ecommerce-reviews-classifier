

class AppendTitleWithMessage():
  def __init__(self):
    return None

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    import numpy as np
    X = X.copy()

    X["review_comment"] = np.where(X["review_comment_title"].isnull()==False,X["review_comment_title"] +" "+ X["review_comment_message"],
                                X["review_comment_message"]) 

    X["review_comment"] = np.where(X["review_comment_message"].isnull(),
                                X["review_comment_title"], X["review_comment"] )
    return X

class SelectAndRenameFeatures():
  def __init__(self):
    return None

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    X = X.copy()

    X = X["review_comment"]
    X.rename("text",inplace=True)

    return X


class FormatText():
  def __init__(self):
    return None

  def fit(self,X,y=None):
    return self


  def format_input(self,text):
    import string
    import unidecode

    for punctuation in string.punctuation:
      text = text.replace(punctuation," ")
      text = text.lower()
      text = unidecode.unidecode(text)
    return text

  def transform(self,X):
    X = X.copy()
    X = X.apply(self.format_input)

    X = X.apply(lambda x: x.split())

    return X


class WordsToIndex():
  def __init__(self,index_dict):
    self.index_dict = index_dict
    return None

  def fit(self,X,y=None):
    return self

  def transform(self,X):
    X = X.copy()
    wordtoidx = self.index_dict
 
    #word to int
    for i in range(len(X)):
      for j in range(len(X.iloc[i])):
        if X.iloc[i][j] in wordtoidx:
          X.iloc[i][j] = wordtoidx[X.iloc[i][j]]
        else:
          X.iloc[i][j] = 1
    
    return X


class ModelPredictor():
  def __init__(self,model,batch_size,device):

    self.batch_size = batch_size
    self.device = device
    self.model = model

  def fit(self,X,y=None):
    return self

  def batches_generator(self,X):
    import numpy as np
    import torch

    num_batches = int (np.ceil(len(X) / self.batch_size))

    batches = []
    for i in range(num_batches):
      x_batch = X[self.batch_size*i:self.batch_size*(i+1)]

      max_len = np.max([len(x) for x in x_batch])
      for j in range(len(x_batch)):
        x = x_batch[j]
        pad = [0] * (max_len - len(x))
        x_batch[j] = pad + x

      x_batch = torch.Tensor(x_batch).long()
      x_batch = x_batch.to(self.device)

      yield x_batch

  def predict(self,X):
    import pandas as pd
    import torch
    import torch.nn as nn

    X = X.copy()

    if isinstance(X,pd.Series) :
      X = X.tolist()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.model.parameters())

    X_iter = lambda: self.batches_generator(X)
    
    outputs = []
    for inputs in X_iter():

      if len(inputs.shape) == 1:
        #single input
        output = self.model(inputs.view(1,-1)).argmax(dim=1)
      else:
        #multiple input
        output = self.model(inputs).argmax(dim=1)

      outputs.extend(output.tolist())

    return outputs