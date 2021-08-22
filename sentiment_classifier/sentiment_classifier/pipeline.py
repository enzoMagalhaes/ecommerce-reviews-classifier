from sklearn.pipeline import Pipeline
from sentiment_classifier.processing import data_management as dm 
from sentiment_classifier.config import config
from sentiment_classifier.processing import preprocessors as p 

nlp_pipe = Pipeline([
    ("append texts",p.AppendTitleWithMessage()),
    ("select text feature",p.SelectAndRenameFeatures()),
    ("format text",p.FormatText()),
    ("transform words to int",p.WordsToIndex(index_dict=dm.get_index_file())),
    ("model",p.ModelPredictor(model=dm.load_model(config.DEVICE),batch_size=32,device=config.DEVICE))
])