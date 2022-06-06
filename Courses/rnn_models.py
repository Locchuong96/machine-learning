import math 
import numpy as np 
import matplotlib.pyplot as plt 

# define activation functions
# sigmoid function get value from 0~1
def sigmoid(x):
    return 1/(1+np.exp(-x))

# tanh function get value from -1~1
def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


class GRU():
    def __init__(self,hidden_dim=100,seq_len=50,input_dim = 1,output_dim = 1):
        self.hidden_dim = hidden_dim 
        self.seq_len = seq_len 
        self.input_dim = input_dim 
        self.output_dim = output_dim  
        # for update gates
        self.U_u = np.random.rand(hidden_dim,seq_len)
        self.W_u = np.random.rand(hidden_dim,hidden_dim)
        self.b_u = np.random.rand(hidden_dim,1)
        # for relevant gates
        self.U_r = np.random.rand(hidden_dim,seq_len)
        self.W_r = np.random.rand(hidden_dim,hidden_dim)
        self.b_r = np.random.rand(hidden_dim,1)
        # for current value
        self.U_h = np.random.rand(hidden_dim,seq_len)
        self.W_h = np.random.rand(hidden_dim,hidden_dim)
        self.b_h = np.random.rand(hidden_dim,1)
        # for output dim
        self.V = np.random.rand(output_dim,hidden_dim)
        self.b_y = np.random.rand(output_dim,1)
       
    def forward_pass(self,x):
        layers = [] 
        h_prev = np.zeros((self.hidden_dim,1))
        seq_len = x.shape[0]
        for t in range(seq_len):
            new_input = np.zeros((seq_len,self.input_dim))
            new_input[t] = x[t]
            # updated gate
            u_t = sigmoid(self.U_u @ new_input + self.W_u @ h_prev + self.b_u)
            # revelant gate
            r_t = sigmoid(self.U_r @ new_input + self.W_r @ h_prev + self.b_r)
            # tilde h
            h_til = tanh(self.U_h @ new_input + self.W_h @ (r_t * h_prev) + self.b_h)
            # h
            h = (1-u_t)* h_prev + u_t * h_til
            # output value
            y_hat = self.V@h + self.b_y
            # collect h_prev,h_til,h,u,r
            layers.append({'h_prev': h_prev,'h_til': h_til,'h':h,'u':u_t,'r':r_t})
            # update h
            prev_h = h
        return layers,y_hat
    
    def calc_loss(self,X,Y):
        loss = 0.0
        n_samples = Y.shape[0] # number of sample
        for i in range(n_samples):
        	y = Y[i]
        	_,y_hat = self.forward_pass(X[i])
        	loss += (y - y_hat)**2
        loss = 1/(2*n_samples)*np.float(loss)
        return loss
    
    def predict(self,X):
        preds= []
        n_samples = X.shape[0] # number of sample
        for i in range(n_samples):
            x = X[i]
            _,y_hat = layers,y_hat = self.forward_pass(x)
            preds.append(y_hat)
        # convert to numpy array
        preds = np.array(preds)
        preds = np.squeeze(preds)
        return preds
    
    def bptt(self,x,y,layers,y_hat,min_val = -10,max_val =10):
        # init matrices h_til = u_t = r_t = (100,1) = (hidden_dim,1)
        dU_u = np.zeros(self.U_u.shape) # (100,50) = (hidden_dim,seq_len)
        dW_u = np.zeros(self.W_u.shape) # (100,100) = (hidden_dim,hidden_dim)
        db_u = np.zeros(self.b_u.shape) # (100,1) = (hidden_dim,1)
        dU_r = np.zeros(self.U_r.shape) # (100,50) = (hidden_dim,seq_len)
        dW_r = np.zeros(self.W_r.shape) # (100,100) = (hidden_dim,hidden_dim)
        db_r = np.zeros(self.b_r.shape) # (100,1) = (hidden_dim,1)
        dU_h = np.zeros(self.U_h.shape) # (100,50) = (hidden_dim,seq_len)
        dW_h = np.zeros(self.W_h.shape) # (100,100) = (hidden_dim,hidden_dim)
        db_h = np.zeros(self.b_h.shape) # (100,1) = (hidden_dim,1)
        dV = np.zeros(self.V.shape)     # (1,100) = (output_dim,hidden_dim)
        db_y = np.zeros(self.b_y.shape) # (1,1) = (1,output_dim)
        
        # dLdy
        dLdy = y - y_hat # (1,1)
        # dLdh
        dLdh = self.V.T @ dLdy  # (100,1)x(1,1) = (100,1)
        # seq_lenght
        for t in range(x.shape[0]):
            # get current timestep input
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            # dV
            dV += dLdy @ np.transpose(layers[t]['h']) # (1,1)x(1,100) = (1,100)
            # db_y
            db_y += dLdy # (1,1)
            # dLdh_til - content-state
            dLdh_til = dLdh*layers[t]['u'] # (100,1)*(100,1)
            # dLdr - reset gate
            dLdr = (1-layers[t]['h_til']**2)*(self.W_r@layers[t]['h_prev']) # (100,1)*[(100,100)x(100,1)]
            #print(f'dLdr {dLdr.shape}')
            # dLdu - update gate
            dLdu = dLdh*(-layers[t]['h_prev'] + layers[t]['h_til'])
            #print(f'dLdu {dLdu.shape}')
            # dldU_u
            dU_h += dLdh_til @ new_input.T # (100,1)x(1,50)
            # dLdW_u
            dW_h += dLdh_til @ layers[t]['h_prev'].T # (100,1)x(1,100)
            # dLdb_u
            db_h += dLdh_til
            # dldU_u
            dU_u += dLdu @ new_input.T # (100,1)x(1,50)
            # dLdW_u
            dW_u += dLdu @ layers[t]['h_prev'].T # (100,1)x(1,100)
            # dLdb_u
            db_u += dLdu
            # dldU_r
            dU_r += dLdr @ new_input.T # (100,1)x(1,50)
            # dLdW_r
            dW_r += dLdr @ layers[t]['h_prev'].T # (100,1)x(1,100)
            # dLdb_r
            db_r += dLdr
            
        db_y = db_y/t
        db_h = db_h/t
        db_u = db_u/t
        db_r = db_r/t
        
        #take care for exploding gradients
        if dV.max() > max_val:
            dV[dV > max_val] = max_val
        if dV.min() < min_val:
            dV[dV < min_val] = min_val
        
        if db_y.max() > max_val:
            db_y[db_y > max_val] = max_val
        if db_y.min() < min_val:
            db_y[db_y < min_val] = min_val
            
        if dU_h.max() > max_val:
            dU_h[dU_h > max_val] = max_val
        if dU_h.min() < min_val:
            dU_h[dU_h < min_val] = min_val
            
        if dW_h.max() > max_val:
            dW_h[dW_h > max_val] = max_val
        if dW_h.min() < min_val:
            dW_h[dW_h < min_val] = min_val

        if db_h.max() > max_val:
            db_h[db_h > max_val] = max_val
        if db_h.min() < min_val:
            db_h[db_h < min_val] = min_val
        
        if dU_u.max() > max_val:
            dU_u[dU_u > max_val] = max_val
        if dU_u.min() < min_val:
            dU_u[dU_u < min_val] = min_val
            
        if dW_u.max() > max_val:
            dW_u[dW_u > max_val] = max_val
        if dW_u.min() < min_val:
            dW_u[dW_u < min_val] = min_val

        if db_u.max() > max_val:
            db_u[db_u > max_val] = max_val
        if db_u.min() < min_val:
            db_u[db_u < min_val] = min_val
            
        if dU_r.max() > max_val:
            dU_r[dU_r > max_val] = max_val
        if dU_r.min() < min_val:
            dU_r[dU_r < min_val] = min_val
            
        if dW_r.max() > max_val:
            dW_r[dW_r > max_val] = max_val
        if dW_r.min() < min_val:
            dW_r[dW_r < min_val] = min_val

        if db_r.max() > max_val:
            db_r[db_r > max_val] = max_val
        if db_r.min() < min_val:
            db_r[db_r < min_val] = min_val
            
        
        return dU_u,dW_u,db_u,dU_r,dW_r,db_r,dU_h,dW_h,db_h,dV,db_y
    
    def train(self,X,Y,epochs,learning_rate,predict = True,verbose = True):
        # storge loss
        losses = []
        for epoch in range(epochs):
            
            loss = self.calc_loss(X,Y)
            losses.append(loss)
            if verbose: print(f'epoch: {epoch} loss: {loss}')
            
            for i in range(X.shape[0]):
                x = X[i]
                y = Y[i]
                # forward pass
                layers,y_hat = layers,y_hat = self.forward_pass(x)
                # backward pass
                dU_u,dW_u,db_u,dU_r,dW_r,db_r,dU_h,dW_h,db_h,dV,db_y = self.bptt(x,y,layers,y_hat)
                # gradient descent
                self.U_u += dU_u*learning_rate
                self.W_u += dW_u*learning_rate
                self.b_u += db_u*learning_rate
                self.U_r += dU_r*learning_rate
                self.W_r += dW_r*learning_rate
                self.b_r += db_r*learning_rate
                self.U_h += dU_h*learning_rate
                self.W_h += dW_h*learning_rate
                self.b_h += db_h*learning_rate
                self.V += dV*learning_rate
                self.b_y += db_y*learning_rate
                
            if predict:
                preds = self.predict(X)
                plt.plot(preds,label = 'pred')
                plt.plot(Y,label = 'ground-truth')
                plt.title('fater training')
                plt.legend()
                plt.show()
                    
        return losses