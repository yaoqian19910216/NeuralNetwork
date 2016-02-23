################## FunctionGradient ###################
import numpy as np 

def SoftMaxFunc(inputValue):
	return np.exp(inputValue) / sum(np.exp(inputValue))

def tanh(inputValue):
	return np.tanh(inputValue)

def tanhGradient(inputValue):
	return 1 - np.multiply(tanh(inputValue),tanh(inputValue))

def sigmoid(inputValue):
	return 1.0 / (1 + np.exp(-inputValue))

def sigmoidGradient(inputValue):
	return np.multiply(sigmoid(inputValue), 1 - sigmoid(inputValue))

def ReLu(inputValue):
	return np.maximum(0, inputValue)

def ReLuGradient(inputValue):
	slope = np.ones(inputValue.shape)
	slope[inputValue <= 0] = 0
	return slope
