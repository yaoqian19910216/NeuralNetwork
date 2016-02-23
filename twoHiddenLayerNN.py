################ twoHiddenLayerNN.py ################
import numpy as np 
from FunctionGradient import SoftMaxFunc, tanh, tanhGradient, sigmoid, sigmoidGradient, ReLu, ReLuGradient
from DataProcess import load_mnist
import matplotlib.pyplot as plt
import timeit
from multilayerNN import SoftmaxTopLayer,  HiddenLayer, test_MLP

class twoHiddenLayerPerceptron(object):
	def __init__(self, rng, n_in, n_hidden1, n_hidden2, n_out, 
			activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-5), regularization = 0, momentum = 0):
		self.labelRange = np.array(range(10))

		self.hiddenLayer1 = HiddenLayer(
			rng = rng, n_in = n_in, n_out = n_hidden1 + 1, activation = activation, activationGradient = activationGradient
		)

		self.hiddenLayer2 = HiddenLayer(
			rng = rng, n_in = n_hidden1 + 1, n_out = n_hidden2 + 1, activation = activation, activationGradient = activationGradient
		)

		self.softmaxTopLayer = SoftmaxTopLayer(
			rng = rng, n_in = n_hidden2 + 1, n_out = n_out
		)


		self.params = [self.hiddenLayer1.params] + [self.hiddenLayer2.params] + [self.softmaxTopLayer.params]
		self.learningRate = learningRate
		self.regularization = regularization
		self.momentum = momentum

	def forwardPropagate(self, inputValue, output):
		outputVec = 1*(self.labelRange == output)

		#propagate to the hidden layer, bias corresponds to a unit with constant ouput 1
		self.hiddenLayer1.inputValue = inputValue
		self.hiddenLayer1.linearOutput = np.dot(self.hiddenLayer1.inputValue, self.hiddenLayer1.W)
		self.hiddenLayer1.y = self.hiddenLayer1.activation(self.hiddenLayer1.linearOutput)
		self.hiddenLayer1.y[0] = 1
		self.hiddenLayer1.yPrime = self.hiddenLayer1.activationGradient(self.hiddenLayer1.linearOutput)
		self.hiddenLayer1.yPrime[0] = 0

		#propagate to the  second hidden layer, bias corresponds to a unit with constant ouput 1
		self.hiddenLayer2.inputValue = self.hiddenLayer1.y
		self.hiddenLayer2.linearOutput = np.dot(self.hiddenLayer2.inputValue, self.hiddenLayer2.W)
		self.hiddenLayer2.y = self.hiddenLayer2.activation(self.hiddenLayer2.linearOutput)
		self.hiddenLayer2.y[0] = 1
		self.hiddenLayer2.yPrime = self.hiddenLayer2.activationGradient(self.hiddenLayer2.linearOutput)
		self.hiddenLayer2.yPrime[0] = 0

		#propagate to the top layer, bias corresponds to a unit with constant ouput 1
		self.softmaxTopLayer.inputValue = self.hiddenLayer2.y
		self.softmaxTopLayer.linearOutput = np.dot(self.softmaxTopLayer.inputValue, self.softmaxTopLayer.W)
		self.softmaxTopLayer.y = SoftMaxFunc(self.softmaxTopLayer.linearOutput)
		self.softmaxTopLayer.predict = np.argmax(self.softmaxTopLayer.y)

		self.softmaxTopLayer.delta = outputVec - self.softmaxTopLayer.y
		self.softmaxTopLayer.weightGradient = self.softmaxTopLayer.weightGradient - \
											np.outer(self.softmaxTopLayer.inputValue,self.softmaxTopLayer.delta)


	def backwardPropagate(self):
		self.hiddenLayer2.delta = np.multiply(self.hiddenLayer2.yPrime, 
			np.dot(self.softmaxTopLayer.W, self.softmaxTopLayer.delta))

		self.hiddenLayer2.weightGradient = self.hiddenLayer2.weightGradient - \
			np.outer(self.hiddenLayer2.inputValue, self.hiddenLayer2.delta)

		self.hiddenLayer1.delta = np.multiply(self.hiddenLayer1.yPrime, 
			np.dot(self.hiddenLayer2.W, self.hiddenLayer2.delta))

		self.hiddenLayer1.weightGradient = self.hiddenLayer1.weightGradient - \
			np.outer(self.hiddenLayer1.inputValue, self.hiddenLayer1.delta)


	def updateWeight(self):
		preW = self.softmaxTopLayer.W
		self.softmaxTopLayer.W = preW - \
			self.learningRate*self.softmaxTopLayer.weightGradient + \
			2*self.regularization*preW + self.momentum*self.softmaxTopLayer.deltaW
		self.softmaxTopLayer.deltaW = self.softmaxTopLayer.W - preW



		preHiddenW2 = self.hiddenLayer2.W
		self.hiddenLayer2.W = preHiddenW2 - \
			self.learningRate*self.hiddenLayer2.weightGradient + \
			2*self.regularization*preHiddenW2 + self.momentum*self.hiddenLayer2.deltaW
		self.hiddenLayer2.deltaW = self.hiddenLayer2.W - preHiddenW2

		preHiddenW1 = self.hiddenLayer1.W
		self.hiddenLayer1.W = preHiddenW1 - \
			self.learningRate*self.hiddenLayer1.weightGradient + \
			2*self.regularization*preHiddenW1 + self.momentum*self.hiddenLayer1.deltaW
		self.hiddenLayer1.deltaW = self.hiddenLayer1.W - preHiddenW1


	def accuracy(self, INPUT, OUTPUT):
		count = 0
		for inputValue, output in zip(INPUT,OUTPUT):
			linearOutputHidden1 = np.dot(inputValue, self.hiddenLayer1.W)
			yHidden1 = self.hiddenLayer1.activation(linearOutputHidden1)
			yHidden1[0] = 1

			linearOutputHidden2 = np.dot(yHidden1, self.hiddenLayer2.W)
			yHidden2 = self.hiddenLayer2.activation(linearOutputHidden2)
			yHidden2[0] = 1

			linearOutput = np.dot(yHidden2, self.softmaxTopLayer.W)
			y = SoftMaxFunc(linearOutput)
			predict = np.argmax(y)

			if predict == output:
				count = count + 1

		return (float(count) / len(OUTPUT))


if __name__ == '__main__':
	allTrain = load_mnist(dataset="training", path='../')

	trainImage = allTrain[0][0:49999,:]
	trainLabel = allTrain[1][0:49999]

	validImage = allTrain[0][50000:,:]
	validLabel = allTrain[1][50000:]

	allTest = load_mnist(dataset="testing",path='../')
	testImage = allTest[0]
	testLabel = allTest[1]

	rng = np.random.RandomState(1234)

	#(D)
	MLP = twoHiddenLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden1 = 43, n_hidden2 = 43, n_out = 10, 
		activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-5))
	test_MLP(MLP, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"twoHiddenLayer.png")

	













		



