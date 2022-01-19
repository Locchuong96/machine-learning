'''
Update: 12/1/22
Author: LocChuong96
Organization: github.com/Iteam1
Description: activation function for Neural Network
'''
import numpy as np

def sigmoid(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return x(1-x)

def identity(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return x

def identity_derivative(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return np.ones(x.shape)

def tanh(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return np.tanh(x) # 2/(1+np.exp(-2*x)) - 1

def tanh_derivative(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return 1-np.power(np.tanh(x),2)

def arctan(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return np.arctan(x)

def arctan_derivative(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return 1/(np.power(x,2)+1)

def softplus(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return np.log(1 + np.exp(x))

def softplus_derivative(x):
	'''
	x dimension:
		- 0-D: a number
		- 1-D: a vector
		- 2-D: a martix
		- 3-D: a tensor
		- n-D: i have no idea what is x too...
	result's shape equal with x's shape 
	'''
	return 1/(1 + np.exp(-x))



