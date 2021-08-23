from sentiment_classifier.processing import preprocessors as p 
from sentiment_classifier.processing import data_management as dm
from sentiment_classifier.config import config

def test_AppendTitleWithMessage(clean_df_sample):
	df = clean_df_sample

	transformer = p.AppendTitleWithMessage()
	transformed_df = transformer.transform(df)

	assert transformed_df['review_comment_message'].isnull().sum() == 0
	assert len(transformed_df) == len(clean_df_sample)
	assert isinstance(transformed_df['review_comment_message'].iloc[0],str)


def test_SelectAndRenameFeatures(clean_df_sample):
	df = clean_df_sample

	transformer = p.AppendTitleWithMessage()
	transformed_df = transformer.transform(df)

	transformer = p.SelectAndRenameFeatures()
	transformed_df = transformer.transform(transformed_df)

	import pandas as pd

	assert isinstance(transformed_df,pd.Series)
	assert len(transformed_df) == len(clean_df_sample)
	assert isinstance(transformed_df.iloc[0],str) 



def test_FormatText(clean_df_sample):
	df = clean_df_sample

	transformer = p.AppendTitleWithMessage()
	transformed_df = transformer.transform(df)

	transformer = p.SelectAndRenameFeatures()
	transformed_df = transformer.transform(transformed_df)

	transformer = p.FormatText()
	transformed_df = transformer.transform(transformed_df)

	import pandas as pd

	assert isinstance(transformed_df,pd.Series)
	assert len(transformed_df) == len(clean_df_sample)
	assert isinstance(transformed_df.iloc[0],list)	 
	assert isinstance(transformed_df.iloc[0][0],str)



def test_WordsToIndex(clean_df_sample):
	df = clean_df_sample

	transformer = p.AppendTitleWithMessage()
	transformed_df = transformer.transform(df)

	transformer = p.SelectAndRenameFeatures()
	transformed_df = transformer.transform(transformed_df)

	transformer = p.FormatText()	
	transformed_df = transformer.transform(transformed_df)

	transformer = p.WordsToIndex(index_dict=dm.get_index_file())
	transformed_df = transformer.transform(transformed_df)

	import pandas as pd

	assert isinstance(transformed_df,pd.Series)
	assert len(transformed_df) == len(clean_df_sample)
	assert isinstance(transformed_df.iloc[0],list)	 
	assert isinstance(transformed_df.iloc[0][0],int)



def test_ModelPredictor(clean_df_sample):
	df = clean_df_sample

	transformer = p.AppendTitleWithMessage()
	transformed_df = transformer.transform(df)

	transformer = p.SelectAndRenameFeatures()
	transformed_df = transformer.transform(transformed_df)

	transformer = p.FormatText()	
	transformed_df = transformer.transform(transformed_df)

	transformer = p.WordsToIndex(index_dict=dm.get_index_file())
	transformed_df = transformer.transform(transformed_df)

	predictor = p.ModelPredictor(model=dm.load_model(config.DEVICE),batch_size=32,device=config.DEVICE)
	predictions = predictor.predict(transformed_df)


	assert isinstance(predictions,list)
	assert len(predictions) == len(clean_df_sample)
	assert isinstance(predictions[0],int)

