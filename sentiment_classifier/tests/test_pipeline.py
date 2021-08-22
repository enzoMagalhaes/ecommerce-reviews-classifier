from sentiment_classifier.predict import make_prediction
from sentiment_classifier.processing import data_management as dm

def test_prediction():
	df = dm.load_dataset()

	df = df[df['review_comment_message'].isnull()==False]

	predictions = make_prediction(df.iloc[0:5])

	#map predictions
	import pandas as pd
	predictions = pd.Series(predictions)
	predictions = predictions.map({0:'positive',1:'neutral',2:'negative'})
	predictions = predictions.tolist()

	targets = df['review_score'].iloc[0:5]
	targets = targets.map({5:'positive',4:'positive',3:'neutral',2:'negative',1:'negative'})
	targets = targets.tolist()

	print(f'targets: {targets} , predictions: {predictions}')

	
if __name__ == '__main__':
	test_prediction()