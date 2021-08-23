import pytest
from sentiment_classifier.config import config 
from sentiment_classifier.processing import data_management as dm 

@pytest.fixture()
def df_sample():
	df = dm.load_dataset()

	return df.iloc[0:10]

@pytest.fixture()
def clean_df_sample():
	df = dm.load_dataset()
	df = df[df['review_comment_message'].isnull()==False]

	return df.iloc[0:10]

@pytest.fixture()
def json_sample():
	
	sample = open("tests/json_test_sample.json")

	import json
	data = json.load(sample)

	sample.close()
	return data

@pytest.fixture()
def single_json_sample():
	
	sample = open("sentiment_classifier/datasets/sample_input.json",)

	import json
	data = json.load(sample)

	sample.close()
	return data