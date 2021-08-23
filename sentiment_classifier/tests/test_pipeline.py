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

def test_model_quality(clean_df):
	# dropping lines which review_comment message is ' ' (they would be filtered by validate_inputs)
	clean_df = clean_df.reset_index()
	clean_df = clean_df.drop(index=[11471, 15504, 17278, 19024, 31778, 34506, 36738, 37365])

	targets = clean_df['review_score']
	#map targets for 0:'positive' 1:'neutral' 2:'negative'
	targets = targets.map({5:0,4:0,3:1,2:2,1:2})

	results = make_prediction(clean_df)
	predictions = results['predictions']

	from sklearn.metrics import accuracy_score

	acc = accuracy_score(targets,predictions)

	print(acc)

	assert acc >= 0.86
	assert results['errors'] == []


