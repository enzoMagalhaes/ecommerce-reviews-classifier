from sentiment_classifier.predict import make_prediction
from sentiment_classifier.processing import data_management as dm

def test_prediction():
	df = dm.load_dataset()

	df = df[df['review_comment_message'].isnull()==False]

	results = make_prediction(df.iloc[0:5])
	predictions = results['predictions']


	#map predictions
	import pandas as pd
	predictions = pd.Series(predictions)
	predictions = predictions.map({0:'positive',1:'neutral',2:'negative'})
	predictions = predictions.tolist()

	targets = df['review_score'].iloc[0:5]
	targets = targets.map({5:'positive',4:'positive',3:'neutral',2:'negative',1:'negative'})
	targets = targets.tolist()

	print(f'targets: {targets} , predictions: {predictions}')


def test_validation():
	df = dm.load_dataset()

	df = df[df['review_comment_message'].isnull()==False]

	#test selected_features
	# df = df[['review_comment_message','review_id']]

	#test typeerror
	# df['review_comment_message'] = 1

	#test NAN filter
	#	DOES NOT WORK !!!!!!!
	df['review_comment_message'] = None

	results = make_prediction(df.iloc[0:1])
	predictions = results['predictions']
	errors = results['errors']

	print(f'predictions: {predictions} , errors: {errors}')


	
if __name__ == '__main__':
	# test_prediction()
	test_validation()