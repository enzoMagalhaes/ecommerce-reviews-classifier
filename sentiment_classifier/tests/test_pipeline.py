from sentiment_classifier.predict import make_prediction
from sentiment_classifier.processing import data_management as dm

def test_prediction(df_sample):
	df = df_sample

	df = df[df['review_comment_message'].isnull()==False]

	results = make_prediction(df)
	predictions = results['predictions']


	assert len(predictions) == len(df)
	assert isinstance(predictions[0],int)

def test_single_input_prediction(df_sample):
	df = df_sample

	df = df[df['review_comment_message'].isnull()==False]
	df = df.iloc[0:1]

	results = make_prediction(df)
	predictions = results['predictions']


	assert len(predictions) == len(df)
	assert isinstance(predictions[0],int)


def test_json_prediction(json_sample):

	results = make_prediction(json_sample)
	predictions = results['predictions']


	assert len(predictions) == len(json_sample)
	assert isinstance(predictions[0],int)

def test_json_single_input_prediction(single_json_sample):

	results = make_prediction(single_json_sample)
	predictions = results['predictions']


	assert len(predictions) == 1
	assert isinstance(predictions[0],int)



