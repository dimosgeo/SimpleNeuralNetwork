# #https://www.youtube.com/watch?v=tMrbN67U9d4
# import numpy as np

# np.random.seed(0)
# # X=[[1,2,3,2.5],
# # 	[2.0,5.0,-1.0,2.0],
# # 	[-1.5,2.7,3.3,-0.8]]
# # X=[[0,0,0],
# # 	[0,1,0],
# # 	[1,0,0],
# # 	[1,1,1]]

# X=[[6,40,0],
# 	[5,40,1],
# 	[8,80,0],
# 	[1,60,1],
# 	[2,40,1],
# 	[2,80,0]]

# class Layer_Dence:
# 	def __init__(self,n_inputs,n_neurons):
# 		self.weights=0.1*np.random.randn(n_inputs,n_neurons)
# 		self.biases=np.zeros((1,n_neurons))
# 	def forward(self,inputs):
# 		self.output=np.dot(inputs,self.weights)+self.biases
# 	# def update_weights(self,real_out):
# 	# 	self.updated_weights

# class Activation_ReLU:
# 	def forward(self,inputs):
# 		self.output=np.maximum(0,inputs)
# 		# self.output=inputs
# 		# self.output=1.0 / (1.0 + np.exp(-inputs))
	
# # Calculate the derivative of an neuron output
# def transfer_derivative(output):
# 	return output * (1.0 - output)

# layer1=Layer_Dence(3,5)
# #print(np.array(X).T)
# #print(layer1.weights)
# #print(layer1.biases)
# activation1=Activation_ReLU()
# layer2=Layer_Dence(5,1)
# activation2=Activation_ReLU()

# for i in range(100):
# 	layer1.forward(X)
# 	activation1.forward(layer1.output)
# 	layer2.forward(activation1.output)
# 	activation2.forward(layer2.output)
# 	#print(activation2.output)

# print(np.array(X).T[2]-np.array(activation2.output).T)

# 		


import numpy as np
# X = (hours sleeping, hours studying), y = test score of the student
Q = np.array(([6, 40], [5, 40], [8, 80],[1,60],[2,40],[2,80]), dtype=float)
X = np.array(([6, 40], [5, 40], [8, 80],[1,60],[2,40],[2,80]), dtype=float)
y = np.array(([0], [1], [0],[1],[1],[0]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) #maximum of X array
#y = y/100 # maximum test score is 100

class NeuralNetwork(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
        
    def feedForward(self, X):
        #forward propogation through the network
        self.z = np.dot(X, self.W1) #dot product of X (input) and first set of weights (3x2)
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden layer (z2) and second set of weights (3x1)
        output = self.sigmoid(self.z3)
        self.output = output
        return output
        
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def backward(self, X, y, output):
        #backward propogate through the network
        self.output_error = y - output # error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv=True)
        
        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) # adjusting second set (hidden -> output) weights
        
    def train(self, X, y):
        output = self.feedForward(X)
        self.backward(X, y, output)
        
NN = NeuralNetwork()

for i in range(1000): #trains the NN 1000 times
    if (i % 1 == 0):
        print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
    NN.train(X, y)
        
print("Input: " + str(X))
print("Actual Output: " + str(y))
print("Loss: " + str(np.mean(np.square(y - NN.feedForward(X)))))
print("\n")
print("Predicted Output: " + str(NN.feedForward(X)))

Z = np.array(([3, 10]), dtype=float)
Z = Z/np.amax(Q, axis=0) #maximum of X array
print(Z)
NN.feedForward(Z)
print(NN.output)