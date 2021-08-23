from sentiment_classifier.config import config
from sentiment_classifier.processing import data_management as dm


def validate_inputs(df):
	df = df.copy()
	
	errors = []

	#check individual feature errors
	for feature in config.SELECTED_FEATURES:

		# check if input has selected features 
		if feature not in df.columns:
			error = f'BadInputError: input does not have {feature} column'
			errors.append(error)
			continue

		# check if input feature types are the expected types
		sample_input = dm.load_sample_input()
		if df[feature].dtype != sample_input[feature].dtype:
			error = f'TypeError: input {feature} type is ({sample_input[feature].dtype}) got -> ({df[feature].dtype})'
			errors.append(error)
			continue

	if len(errors):
		return (None , errors)



	# check/filter if input has Nans
	from sentiment_classifier.processing.preprocessors import AppendTitleWithMessage
	import pandas as pd
	transformer = AppendTitleWithMessage()
	transformed_df = transformer.transform(df)

	for i in range(len(transformed_df)):
		if transformed_df["review_comment"].iloc[i] == ' ' or pd.isnull(transformed_df["review_comment"].iloc[i]):
		  #filter the row
			df.drop(index=i,inplace=True)
				
			error = f'NullError: input no: {i} is NULL , input filtered'
			errors.append(error)

	return (df , errors)






