import tensorflow as tf 
import numpy as np

class NeuralNetwork:
    '''
    Neural Network with only sigmoid activation function
    
    Arguments:
    layers --- the list of node number each layer (example: [2,2,1])
    lr     --- the learning_rate (default = 0.1)
    
    Return:
    Object neural network
    '''
    def __init__(self,layers,lr = 0.1):
        '''
        The initalize function
        
        Arguments:
        layers --- the list of node number each layer
        lr     --- the learning_rate (default = 0.1)

        Return:
        Object neural network
        '''
        self.dim = 2 # the diemension of planar features
        self.layers = [self.dim] + layers # the layer of model
        self.lr = lr # the learning rate
        self.history = {"history":[]}
        # list of W and b
        self.W = [] # the weight
        self.b = [] # the bias
        # initiate value of weight and bias, except the input layer
        for i in range(len(self.layers)-1):
            w_ = np.random.randn(self.layers[i],self.layers[i+1]) # gaussian distribution
            b_ = np.zeros((self.layers[i+1],1)) # zeros at the beginning
            self.W.append(w_)
            self.b.append(b_)
    
    def __repr__(self):
        '''
        Summary model paramters and layers
        
        Arguments:
        None
        
        Return:
        String of summary
        '''
        summary_str = "\n"
        total_param = 0 # total parameters of the model
        for i in range(len(self.layers)):
            if i == 0: # if that is the input layers
                summary_str += f"layer: {i} weight: {0} bias: {0} total: {0}\n"
                total_param += 0 
            else:
                summary_str += f"layer: {i} weight: {self.layers[i-1]*self.layers[i]} bias: {self.layers[i]} total: {self.layers[i]*(self.layers[i-1]+1)}\n"
                total_param += self.layers[i] * (self.layers[i-1]+1) 
        summary_str += f"Total: {total_param}"
        return f"Neural Network [{summary_str}]"
    
    def predict(self,X):
        '''
        Predict the labels (do the feedforward) 
        
        Arguments:
        X -- input or features (n_samples,n_dimensions)
        
        Return:
        X -- predicted labels (n_samples,1)
        '''
        for i in range(0,len(self.layers) - 1):
            X = sigmoid(np.dot(X,self.W[i]) + self.b[i].T)
        return X
    
    def evaluate(self,X,y):
        '''
        Calculate loss between predicted labels and ground truth label
        
        Arguments:
        X -- input or features (n_samples,n_dimensions)
        y -- the ground truth
        
        Return:
        loss -- the loss value
        '''
        y_predict = self.predict(X)
        return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict)))
    
    def fit_partial(self,x,y):
        '''
        Train model each epoch include feedforward and back propagation
        
        Arguments:
        x --- input or features (n_samples,n_dimensions)
        y --- output or labels (n_samples,1)
        
        Return:
        None
        '''
        A = [x] # create the new list of A examples: layers = [2,1] A = [x,a1,a2] 
        # feedforward
        out = A[-1]
        for i in range(0,len(self.layers)-1):
            out = sigmoid(out @ self.W[i] + self.b[i].T)
            A.append(out)
        
        # backprogation
        dA = [ -(y/A[-1] - (1-y)/(1-A[-1])) ] # list of a deriviation, the first is [dA2,]
        dW = []
        db = []
        for i in reversed(range(0,len(self.layers) - 1)):
            dw_ = A[i].T @ (dA[-1] * sigmoid_derivative(A[i+1]))
            db_ =   (np.sum(dA[-1] * sigmoid_derivative(A[i+1]),0)).reshape(-1,1)
            dA_ =          (dA[-1] * sigmoid_derivative(A[i+1])) @ self.W[i].T
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
            
        # invert the order of dW,db [dW1,dW0] -> [dW0,dW1]
        dW = dW[::-1]
        db = db[::-1]
        
        # Gradient descent
        for i in range(0,len(self.layers)-1):
            self.W[i] = self.W[i] - self.lr * dW[i]
            self.b[i] = self.b[i] - self.lr * db[i]
    
    def fit(self,X,y,epochs =20,verbose =1,milestone = 10):
        '''
        Train the model
        
        Arguments:
        X         --- input or features (n_samples,n_dimensions)
        y         --- ground truth label
        epochs    --- the number of time you train your epoch (default = 20)
        verbose   --- decide to print out while training or not (default = 1)
        milestone --- print out the result if current epoch % milestone and verbose is true
        
        Return:
        None
        '''
        self.history['history'] = [] # empty the loss history
        for epoch in range(0,epochs):
            self.fit_partial(X,y)
            loss = self.evaluate(X,y)
            self.history['history'].append(loss)
            if (verbose) and (epoch % milestone == 0):
                print("Epoch {},loss {}".format(epoch,loss))
                        
        if verbose and (epoch % milestone == 0):
            print("Epoch {},loss {}".format(epoch,loss))
                        
