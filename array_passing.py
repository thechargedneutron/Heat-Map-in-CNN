import numpy as np

def return_padded_array(orig_image, non_zero_result_shape, start_index):
	'''
	Parameters
	----------
	orig_image: Original Image of as many dimensions
	non_zero_result_shape: Tuple of result dimensions
	start_index: Again a tuple of indices
	'''
	ans = np.zeros(orig_image.shape)
	ans[start_index[0]:start_index[0]+non_zero_result_shape[0], start_index[1]:start_index[1]+non_zero_result_shape[1], :] = ans[start_index[0]:start_index[0]+non_zero_result_shape[0], start_index[1]:start_index[1]+non_zero_result_shape[1], :]
	return ans

def return_label(input, model, classification):
	pass

def add_probabilities(orig_prob, non_zero_result_shape, start_index, pred_value):
	'''
	Parameters
	----------
	orig_prob: Original Probability matrix of as many dimensions
	non_zero_result_shape: Tuple of result dimensions
	start_index: Again a tuple of indices
	pred_value: The value to be added
	'''
	ans = orig_prob
	ans[start_index[0]:start_index[0]+non_zero_result_shape[0], start_index[1]:start_index[1]+non_zero_result_shape[1]] += pred_value
	return ans