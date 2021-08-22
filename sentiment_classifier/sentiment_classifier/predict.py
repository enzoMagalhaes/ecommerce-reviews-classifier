from sentiment_classifier.config import config
from sentiment_classifier.pipeline import nlp_pipe
import pandas as pd

def make_prediction(X):

	X = pd.DataFrame(X)
	X = X[config.SELECTED_FEATURES]

	# validated_inputs,errors = validate_inputs(X)

	predictions = nlp_pipe.predict(X)

	return predictions