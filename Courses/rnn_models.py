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

class RNN():
    def __init__(self,hidden_dim=100,seq_len=50,input_dim = 1,output_dim = 1,seed = 3454):
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim  = output_dim
        self.U = np.random.uniform(0,1,(hidden_dim,seq_len)) # (100,50)
        self.W = np.random.uniform(0,1,(hidden_dim,hidden_dim)) # (100,100)
        self.V = np.random.uniform(0,1,(output_dim,hidden_dim)) # (1,100)
        self.bh = np.random.uniform(0,1,(hidden_dim,1))
        self.by = np.random.uniform(0,1,(output_dim,1))
        
    def forward_pass(self,x):
        # init a list of dict to storage
        layers = []
        h_prev = np.zeros((hidden_dim,1))
        for t in range(x.shape[0]):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            z = self.U @ new_input + self.W @ h_prev + self.bh
            h = tanh(z)
            y_hat = V @ h + self.by
            layers.append({'h':h,'h_prev':h_prev})
            h_prev = h
        return layers,y_hat
        
    def calc_loss(self,X,Y):
        loss = 0.0
        m = Y.shape[0]
        for i in range(m):
            x,y = X[i],Y[i]
            _,y_hat = self.forward_pass(x)
            loss += (y-y_hat)**2
        loss = 1/(2*m) * np.float(loss)
        return loss

    def predict(self,X):
        preds = []
        m = X.shape[0] # number of samples
        for i in range(m):
            x = X[i]
            _,y_hat = self.forward_pass(x)
            preds.append(y_hat)
        # convert to numpy array
        preds = np.array(preds)
        preds = np.squeeze(preds)
        return preds
    
    def calc_prev_d(self,h,d,W):
        '''
        Calculate the next previous term d after the first term, this function support for bptt function
        Ex: d2 = d3*W*(1-h**2)
        '''
        d_sum = (1-h**2)*d
        return W.T @ d_sum
    
    def bptt(self,x,y,layers,y_hat,bptt_truncate,min_val=-10,max_val=10):
        # differentials at current prediction
        dW = np.zeros(self.W.shape)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        db_h = np.zeros(self.bh.shape)
        db_y = np.zeros(self.by.shape)
        # differentials each timestep
        dW_t = np.zeros(self.W.shape)
        dU_t = np.zeros(self.U.shape)
        dV_t = np.zeros(self.V.shape)
        # diffeentials each backpropagation truncate
        dW_i = np.zeros(self.W.shape)
        dU_i = np.zeros(self.U.shape)
        dV_i = np.zeros(self.V.shape)
        # dLdy
        dLdy = y - y_hat
        # dLdh
        dLdh = self.V.T @ dLdy
        # dLdby
        db_y = dLdy
        for t in range(x.shape[0]):
            # dLdV
            dV_t = dLdy @ np.transpose(layers[t]['h'])
            # first term d = (y-y_hat)V
            d_t = dLdh * (1 - layers[t]['h']**2)
            # dLdbh
            db_h += d_t
            for _ in range(t,max(-1,bptt_truncate-1),-1):
                new_input = np.zeros(x.shape)
                new_input[_] = x[_]
                dU_i = d_t @ new_input.T
                dW_i = d_t @ layers[_]['h_prev'].T
                dU_t += dU_i
                dW_t += dW_i
                # update term d
                d_t = self.calc_prev_d(layers[_]['h_prev'],d_t,self.W)
            dV += dV_t
            dU += dU_t
            dW += dW_t
            # take care of possible exploding gradients
            if dU.max() > max_val:
                dU[dU > max_val] = max_val
            if dV.max() > max_val:
                dV[dV > max_val] = max_val
            if dW.max() > max_val:
                dW[dW > max_val] = max_val

            if dU.min() < min_val:
                dU[dU < min_val] = min_val
            if dV.min() < min_val:
                dV[dV < min_val] = min_val
            if dW.min() < min_val:
                dW[dW < min_val] = min_val
        return dU,dV,dW,db_h,db_y
            
    def train(self,X,Y,epochs,learning_rate,bptt_truncate,min_val,max_val,predict = True,verbose = True):
        # storage lost
        losses = []
        for epoch in range(epochs):
            loss = self.calc_loss(X,Y)
            losses.append(loss)
            title = f'epoch: {epoch} loss: {loss}'
            if verbose: print(title)
            
            for i in range(X.shape[0]):
                x,y = X[i],Y[i]
                # forward pass
                layers,y_hat = self.forward_pass(x)
                dU,dV,dW,db_h,db_y = self.bptt(x,y,layers,y_hat,bptt_truncate,min_val,max_val)
                # SGD
                self.U += learning_rate * dU
                self.W += learning_rate * dW
                self.V += learning_rate * dV
                self.bh += learning_rate * db_h
                self.by += learning_rate * db_y
                
            if predict:
                preds = self.predict(X)
                plt.plot(preds,label = 'pred')
                plt.plot(Y,label = 'ground-truth')
                plt.title(title)
                plt.legend()
                plt.show()
                    
        return losses
        
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
            
        db_y = db_y/t
        db_h = db_h/t
        db_u = db_u/t
        db_r = db_r/t
        
        return dU_u,dW_u,db_u,dU_r,dW_r,db_r,dU_h,dW_h,db_h,dV,db_y
    
    def train(self,X,Y,epochs,learning_rate,min_val,max_val,predict = True,verbose = True):
        # storge loss
        losses = []
        for epoch in range(epochs):
            
            loss = self.calc_loss(X,Y)
            losses.append(loss)
            title = f'epoch: {epoch} loss: {loss}' 
            if verbose: print(title)
            
            for i in range(X.shape[0]):
                x = X[i]
                y = Y[i]
                # forward pass
                layers,y_hat = self.forward_pass(x)
                # backward pass
                dU_u,dW_u,db_u,dU_r,dW_r,db_r,dU_h,dW_h,db_h,dV,db_y = self.bptt(x,y,layers,y_hat,min_val = min_val,max_val = max_val)
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
                plt.title(title)
                plt.legend()
                plt.show()
                    
        return losses