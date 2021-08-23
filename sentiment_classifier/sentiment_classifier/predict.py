from sentiment_classifier.config import config
from sentiment_classifier.pipeline import nlp_pipe
from sentiment_classifier.processing.validation import validate_inputs
from sentiment_classifier.processing import data_management as dm
import pandas as pd

def make_prediction(X):

	# workaround a single json input error
	if type(X) is dict and type(X['review_comment_message']) is str:
		X = pd.DataFrame([X])
	else:
		X = pd.DataFrame(X)

	validated_inputs,errors = validate_inputs(X)

	if validated_inputs is None:
		return {"predictions":None,'errors':errors}


	validated_inputs = validated_inputs[config.SELECTED_FEATURES]
	predictions = nlp_pipe.predict(validated_inputs)

	return {"predictions":predictions,'errors':errors}