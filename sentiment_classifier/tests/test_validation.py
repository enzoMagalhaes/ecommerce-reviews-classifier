from sentiment_classifier.processing.validation import validate_inputs

def test_filters_bad_input(df_sample):

	df = df_sample[['review_comment_message','order_id']]

	validated_inputs,errors = validate_inputs(df)

	assert validated_inputs is None
	assert len(errors) == 1
	assert errors[0] == 'BadInputError: input does not have review_comment_title column'


	df = df_sample[['review_id','order_id']]

	validated_inputs,errors = validate_inputs(df)

	assert validated_inputs is None
	assert len(errors) == 2
	assert errors[0] == 'BadInputError: input does not have review_comment_title column'
	assert errors[1] == 'BadInputError: input does not have review_comment_message column'

def test_filters_type_error(df_sample):

	df = df_sample
	df['review_comment_message'] = 1

	validated_inputs,errors = validate_inputs(df)

	assert validated_inputs is None
	assert len(errors) == 1
	assert errors[0] == 'TypeError: input review_comment_message type is (object) got -> (int64)'

def test_filters_nan(df_sample):

	validated_inputs,errors = validate_inputs(df_sample)

	assert validated_inputs is not None
	assert len(errors) > 0 
	assert errors[0] == 'NullError: input no: 0 is NULL , input filtered'
