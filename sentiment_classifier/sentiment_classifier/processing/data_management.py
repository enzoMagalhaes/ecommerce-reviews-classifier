from sentiment_classifier.config import config

def get_index_file():
    import pickle
    idxfile = open(config.VOCAB_PATH, "rb")
    output = pickle.load(idxfile)
    idxfile.close()

    return output

def load_model(device='cpu'):
   import torch
   from sentiment_classifier.model import RNN

   model = RNN(len(get_index_file()),20,20,0,3)
   model.to(device)

   model.load_state_dict(torch.load(config.WEIGHTS_PATH,map_location=torch.device(device)))
   model.eval()
   return model

def load_dataset(filename='olist_order_reviews_dataset.csv'):
   import pandas as pd
   df = pd.read_csv(config.DATASETS_DIR/filename)
   return df

def load_sample_input(filename='sample_input.json'):
   import pandas as pd 
   import json
   file = open(config.DATASETS_DIR/filename)
   data = json.load(file)

   file.close()

   df = pd.DataFrame([data])

   return df 