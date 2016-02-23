######################## multilayer.py ##################
import numpy as np 
from FunctionGradient import SoftMaxFunc, tanh, tanhGradient, sigmoid, sigmoidGradient, ReLu, ReLuGradient
from DataProcess import load_mnist
import matplotlib.pyplot as plt
import timeit


class SoftmaxTopLayer(object):
	def __init__(self, rng, n_in, n_out):
		self.W = np.asarray( rng.uniform(
			low = -np.sqrt(6.0 / (n_in + n_out)),
			high = np.sqrt(6.0 / (n_in + n_out)),
			size = (n_in, n_out)
			)
		)
		#W(t) - W(t - 1)
		self.deltaW = np.zeros((n_in, n_out))
		
		self.delta = np.zeros(n_out)
		#\partial (E) / \partial (W)
		self.weightGradient = np.zeros((n_in, n_out))
		self.inputValue = np.zeros(n_in)
		self.y = SoftMaxFunc(np.dot(self.inputValue, self.W))
		self.predict = np.argmax(self.y)
		self.params = self.W



class HiddenLayer(object):
	def __init__(self, rng, n_in, n_out, activation, activationGradient):
		self.W = np.asarray( rng.uniform(
			low = -np.sqrt(6.0 / (n_in + n_out)),
			high = np.sqrt(6.0 / (n_in + n_out)),
			size = (n_in, n_out)
			)
		)
		self.W[:,0] = 0
		#W(t) - W(t - 1)
		self.deltaW = np.zeros((n_in,n_out))

		#print self.W
		self.activation = activation
		self.activationGradient = activationGradient
		
		#the derivative at bias unit is zero
		self.delta = np.zeros(n_out)
		self.weightGradient = np.zeros((n_in, n_out))
		self.inputValue = np.zeros(n_in)
		self.linearOutput = np.dot(self.inputValue, self.W)
		#add bias unit
		self.y = activation(self.linearOutput)
		self.y[0] = 1

		self.yPrime = activationGradient(self.linearOutput)
		self.yPrime[0] = 0

		self.params = self.W


class MultiLayerPerceptron(object):
	def __init__(self, rng, n_in, n_hidden, n_out, 
			activation = tanh, activationGradient = tanhGradient, learningRate = 10**(-6), regularization = 0, momentum = 0):
		self.labelRange = np.array(range(10))

		self.hiddenLayer = HiddenLayer(
			rng = rng, n_in = n_in, n_out = n_hidden + 1, activation = activation, activationGradient = activationGradient
		)

		self.softmaxTopLayer = SoftmaxTopLayer(
			rng = rng, n_in = n_hidden + 1, n_out = n_out
		)


		self.params = [self.hiddenLayer.params] + [self.softmaxTopLayer.params]
		self.learningRate = learningRate
		self.regularization = regularization
		self.momentum = momentum

	def forwardPropagate(self, inputValue, output):
		outputVec = 1*(self.labelRange == output)

		#propagate to the hidden layer, bias corresponds to a unit with constant ouput 1
		self.hiddenLayer.inputValue = inputValue
		self.hiddenLayer.linearOutput = np.dot(self.hiddenLayer.inputValue, self.hiddenLayer.W)
		self.hiddenLayer.y = self.hiddenLayer.activation(self.hiddenLayer.linearOutput)
		self.hiddenLayer.y[0] = 1

		self.hiddenLayer.yPrime = self.hiddenLayer.activationGradient(self.hiddenLayer.linearOutput)
		self.hiddenLayer.yPrime[0] = 0

		#propagate to the top layer, bias corresponds to a unit with constant ouput 1
		self.softmaxTopLayer.inputValue = self.hiddenLayer.y
		self.softmaxTopLayer.linearOutput = np.dot(self.softmaxTopLayer.inputValue, self.softmaxTopLayer.W)
		self.softmaxTopLayer.y = SoftMaxFunc(self.softmaxTopLayer.linearOutput)
		self.softmaxTopLayer.predict = np.argmax(self.softmaxTopLayer.y)

		self.softmaxTopLayer.delta = outputVec - self.softmaxTopLayer.y
		self.softmaxTopLayer.weightGradient = self.softmaxTopLayer.weightGradient - \
			np.outer(self.softmaxTopLayer.inputValue,self.softmaxTopLayer.delta)


	def backwardPropagate(self):
		self.hiddenLayer.delta = np.multiply(self.hiddenLayer.yPrime, 
			np.dot(self.softmaxTopLayer.W, self.softmaxTopLayer.delta))

		self.hiddenLayer.weightGradient = self.hiddenLayer.weightGradient - \
			np.outer(self.hiddenLayer.inputValue, self.hiddenLayer.delta)


	def updateWeight(self):
		preW = self.softmaxTopLayer.W
		self.softmaxTopLayer.W = preW - \
			self.learningRate*self.softmaxTopLayer.weightGradient + \
			2*self.regularization*preW + self.momentum*self.softmaxTopLayer.deltaW
		self.softmaxTopLayer.deltaW = self.softmaxTopLayer.W - preW



		preHiddenW = self.hiddenLayer.W
		self.hiddenLayer.W = preHiddenW - \
			self.learningRate*self.hiddenLayer.weightGradient + \
			2*self.regularization*preHiddenW + self.momentum*self.hiddenLayer.deltaW
		self.hiddenLayer.deltaW = self.hiddenLayer.W - preHiddenW 


	def accuracy(self, INPUT, OUTPUT):
		count = 0
		for inputValue, output in zip(INPUT,OUTPUT):
			linearOutputHidden = np.dot(inputValue, self.hiddenLayer.W)
			yHidden = self.hiddenLayer.activation(linearOutputHidden)
			yHidden[0] = 1

			linearOutput = np.dot(yHidden, self.softmaxTopLayer.W)
			y = SoftMaxFunc(linearOutput)
			predict = np.argmax(y)

			if predict == output:
				count = count + 1

		return (float(count) / len(OUTPUT))

def test_MLP(MLP, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,fileName, nEpochs=18):
	batchSize = 500
	numBatch = trainImage.shape[0] / batchSize

	trainingAccuracy = []
	testAccuracy = []
	preValidAccuracy = 0.05

	for k in range(nEpochs):
		start_time = timeit.default_timer()
		for i in range(numBatch):
			#miniBatch
			for j in range(batchSize):
				index = i*batchSize + j
				MLP.forwardPropagate(trainImage[index,:], trainLabel[index])
				MLP.backwardPropagate()

			MLP.updateWeight()
		validAccuracy = MLP.accuracy(validImage, validLabel)

		
		if validAccuracy > 0.95 and (abs(validAccuracy - preValidAccuracy) / preValidAccuracy < 0.0005):
			break
		preValidAccuracy = validAccuracy

		end_time = timeit.default_timer()
		print "one pass takes " + str(end_time - start_time) + 's'
		trainingAccuracy.append(MLP.accuracy(trainImage, trainLabel))
		testAccuracy.append(MLP.accuracy(testImage,testLabel))

	plt.plot(range(1, len(trainingAccuracy) + 1,1), trainingAccuracy,'ro-', linewidth=2)
	plt.plot(range(1, len(testAccuracy) + 1,1), testAccuracy,'b*--', linewidth=2)
	plt.legend(["Training", "Test"],loc = 2 )
	plt.xlabel('iteration')
	plt.ylabel('accuracy')
	plt.title('alpha:' + str(MLP.learningRate) + ", regularization:" + str(MLP.regularization) + \
		", momentum:" + str(MLP.momentum))
	plt.savefig('./' + str(fileName))
	plt.show()
	

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
	MLP = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 50, n_out = 10, 
		activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-6))
	test_MLP(MLP, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"sigmoid.png")

	#(E) /over flow occur
	MLPRegu1 = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 50, n_out = 10, 
		activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-6), regularization = 0.001)
	test_MLP(MLPRegu1, trainImage, trainLabel, validImage, validLabel, testImage, testLabel, "regularization1.png")

	MLPRegu2 = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 50, n_out = 10, 
		activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-6), regularization = 0.0001)
	test_MLP(MLPRegu2, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"regularization2.png")

	#(F)
	MLPMomentum = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 50, n_out = 10, 
		activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-6), momentum = 0.9)
	test_MLP(MLPMomentum, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"momentum.png")

	#(G)
	MLPTanh = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 50, n_out = 10, 
		activation = tanh, activationGradient = tanhGradient, learningRate = 10**(-6))
	test_MLP(MLPTanh, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"tanh.png")

	MLPReLu = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 50, n_out = 10, 
		activation = ReLu, activationGradient = ReLuGradient, learningRate = 10**(-6))
	test_MLP(MLPReLu, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"ReLu.png", nEpochs=14)

	#(H)
	MLPHalf = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 25, n_out = 10, 
		activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-6))
	test_MLP(MLPHalf, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"HalfHiddenUnits.png")

	MLPDouble = MultiLayerPerceptron(rng, n_in = 28*28 + 1, n_hidden = 100, n_out = 10, 
		activation = sigmoid, activationGradient = sigmoidGradient, learningRate = 10**(-6))
	test_MLP(MLPDouble, trainImage, trainLabel, validImage, validLabel, testImage, testLabel,"DoubleHiddenUnits.png")













		



