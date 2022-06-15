import tqdm
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
        # differentials each backpropagation truncate
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
            
            for i in tqdm.tqdm(range(X.shape[0])):
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
    
    def train(self,X,Y,epochs,learning_rate,min_val,max_val,verbose = True):
        # storge loss
        losses = []
        for epoch in range(epochs):
            
            loss = self.calc_loss(X,Y)
            losses.append(loss)
            title = f'epoch: {epoch} loss: {loss}' 
            if verbose: print(title)
            
            for i in tqdm.tqdm(range(X.shape[0])):
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
                    
        return losses
       
class GRU2():
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
       
    def forward_pass(self,x,h_prev):
        u_t = sigmoid(self.U_u @ x + self.W_u @ h_prev + self.b_u)
        # revelant gate
        r_t = sigmoid(self.U_r @ x + self.W_r @ h_prev + self.b_r)
        # tilde h
        h_til = tanh(self.U_h @ x + self.W_h @ (r_t * h_prev) + self.b_h)
        # h
        h = (1-u_t)* h_prev + u_t * h_til
        # output value
        y_hat = self.V@h + self.b_y

        return y_hat,h,h_til,h_prev,u_t,r_t
    
    def calc_loss(self,X,Y):
        loss = 0.0
        n_samples = Y.shape[0] # number of sample
        h_prev = np.zeros((self.hidden_dim,1)) # init the hidden unit
        for i in range(n_samples):
            x,y = X[i],Y[i]
            y_hat,h,h_til,h_pev,u,r = self.forward_pass(x,h_prev) # take hidden unit to calculate then update it
            loss += (y - y_hat)**2
        loss = 1/(2*n_samples)*np.float(loss)
        return loss
    
    def predict(self,X):
        preds= []
        n_samples = X.shape[0] # number of sample
        h_prev = np.zeros((self.hidden_dim,1)) # init the hidden unit
        for i in range(n_samples):
            x = X[i]
            y_hat,h,h_til,h_pev,u,r = self.forward_pass(x,h_prev) # take hidden unit to calculate then update it
            preds.append(y_hat)
        # convert to numpy array
        preds = np.array(preds)
        preds = np.squeeze(preds)
        return preds
    
    def bptt(self,x,y,y_hat,t,layers,dLdh_next,min_val = -10,max_val =10):        
        # dLdy
        dLdy = y - y_hat # (1,1)
        # dLdh
        dLdh = dLdh_next + self.V.T @ dLdy   # (100,1)x(1,1) = (100,1)
        # dV
        dV = dLdy @ np.transpose(layers[t]['h']) # (1,1)x(1,100) = (1,100)
        # db_y
        db_y = dLdy # (1,1)

        # dLdh_til - content-state
        dLdh_til = dLdh*layers[t]['u'] # (100,1)*(100,1)
        # dLdr - reset gate
        dLdr = (1-layers[t]['h_til']**2)*(self.W_r@layers[t]['h_prev']) # (100,1)*[(100,100)x(100,1)]
        #print(f'dLdr {dLdr.shape}')
        # dLdu - update gate
        dLdu = dLdh*(-layers[t]['h_prev'] + layers[t]['h_til'])

        # calculate derivative of hidden_layer for the next step
        dLdh_after = self.W_h @ ( (1-layers[t]['h_til']**2) * dLdh_til) + \
                    self.W_u @ ( layers[t]['u']*(1-layers[t]['u']) * dLdu) + \
                    self.W_r @ ( layers[t]['r']*(1-layers[t]['r']) * dLdr)

        # dldU_u
        dU_h = dLdh_til @ x.T # (100,1)x(1,50)
        # dLdW_u
        dW_h = dLdh_til @ layers[t]['h_prev'].T # (100,1)x(1,100)
        # dLdb_u
        db_h = dLdh_til
        # dldU_u
        dU_u = dLdu @ x.T # (100,1)x(1,50)
        # dLdW_u
        dW_u = dLdu @ layers[t]['h_prev'].T # (100,1)x(1,100)
        # dLdb_u
        db_u = dLdu
        # dldU_r
        dU_r = dLdr @ x.T # (100,1)x(1,50)
        # dLdW_r
        dW_r = dLdr @ layers[t]['h_prev'].T # (100,1)x(1,100)
        # dLdb_r
        db_r = dLdr
        
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
            
        
        return dU_u,dW_u,db_u,dU_r,dW_r,db_r,dU_h,dW_h,db_h,dV,db_y,dLdh_after
    
    def train(self,X,Y,epochs,learning_rate,min_val,max_val,verbose = True):
        # storge loss
        losses = []
        for epoch in range(epochs):
            
            loss = self.calc_loss(X,Y)
            losses.append(loss)
            title = f'epoch: {epoch} loss: {loss}' 
            if verbose: print(title)
            
            # FORWARD PASS
            layers = []
            # init previous state
            h_prev = np.zeros((self.hidden_dim,1))

            for t in range(Y.shape[0]):
                x,y = X[t],Y[t]
                # feed forward
                y_hat,h,h_til,h_pev,u,r = self.forward_pass(x,h_prev)
                layers.append({
                                'h':h,
                                'h_til':h_til,
                                'h_prev':h_prev,
                                'u':u,
                                'r':r,
                                'y_hat':y_hat
                                })

            # BACKWARD PASS
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
        

            dLdh_next = np.zeros((self.hidden_dim,1))

            for t in reversed(range(X.shape[0])):
                dU_u,dW_u,db_u,dU_r,dW_r,db_r,dU_h,dW_h,db_h,dV,db_y,dLdh_after = self.bptt(x,y,y_hat,t,layers,dLdh_next,min_val = min_val,max_val = max_val)
                # update the derivative
                dLdh_next = dLdh_after

                # accumate the your gradients
                dW_u += dW_u
                db_u += db_u
                dU_r += dU_r
                dW_r += dW_r
                db_r += db_r
                dU_h += dU_h
                dW_h += dW_h
                db_h += db_h
                dV += dV
                db_y += db_y


            db_y = db_y/X.shape[0]
            db_h = db_h/X.shape[0]
            db_u = db_u/X.shape[0]
            db_r = db_r/X.shape[0]

            # gradient descent
            self.U_u -= dU_u*learning_rate
            self.W_u -= dW_u*learning_rate
            self.b_u -= db_u*learning_rate
            self.U_r -= dU_r*learning_rate
            self.W_r -= dW_r*learning_rate
            self.b_r -= db_r*learning_rate
            self.U_h -= dU_h*learning_rate
            self.W_h -= dW_h*learning_rate
            self.b_h -= db_h*learning_rate
            self.V -= dV*learning_rate
            self.b_y -= db_y*learning_rate
                    
        return losses

class LSTM():
    def __init__(self,seq_len = 50,hidden_dim = 100,input_dim = 1,output_dim = 1):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        # init matrices
        # input gate
        self.W_i = np.zeros((hidden_dim,hidden_dim + seq_len))
        self.b_i = np.zeros((hidden_dim,1))
        # forget gate
        self.W_f = np.zeros((hidden_dim,hidden_dim + seq_len))
        self.b_f = np.zeros((hidden_dim,1))
        # output gate
        self.W_o = np.zeros((hidden_dim,hidden_dim + seq_len))
        self.b_o = np.zeros((hidden_dim,1))
        # candidate cell state
        self.W_c = np.zeros((hidden_dim,hidden_dim + seq_len))
        self.b_c = np.zeros((hidden_dim,1))
        # predict state
        self.V = np.zeros((output_dim,hidden_dim))
        self.b_y = np.zeros((1,1))
        
    def forward_pass(self,h_prev,C_prev,x):
        # row-stack h_prev and x
        z = np.row_stack((h_prev,x))
        # forget gate
        f_t = sigmoid(self.W_f @ z + self.b_f)
        # input gate
        i_t = sigmoid(self.W_i @ z + self.b_i)
        # output gate
        o_t = sigmoid(self.W_o @ z + self.b_o)
        # candidate cell state
        C_til = tanh(self.W_c @ z + self.b_c)
        # cell state
        C_t = f_t * C_prev + i_t * C_til
        # hidden state
        h_t = o_t * tanh(C_t)
        # predict output
        y_hat = self.V @ h_t + self.b_y
        # update previous cell state and hidden state
        h_prev = h_t
        C_prev = C_t
        return h_prev,C_prev,f_t,i_t,o_t,C_til,C_t,h_t,y_hat
    
    def calc_loss(self,X,Y):
        loss = 0.0
        m = Y.shape[0]
        # init previous state
        h_prev = np.zeros((self.hidden_dim,1))
        C_prev = np.zeros((self.hidden_dim,1))
        for i in range(m):
            x,y = X[i],Y[i]
            h_prev,C_prev,f_t,i_t,o_t,C_til,C_t,h_t,y_hat = self.forward_pass(h_prev,C_prev,x)
            loss += (y-y_hat)**2
        loss = 1/(2*m) * np.float(loss)
        return loss

    def predict(self,X):
        preds = []
        m = X.shape[0] # number of samples
        h_prev = np.zeros((self.hidden_dim,1))
        C_prev = np.zeros((self.hidden_dim,1))
        for i in range(m):
            x = X[i]
            h_prev,C_prev,f_t,i_t,o_t,C_til,C_t,h_t,y_hat = self.forward_pass(h_prev,C_prev,x)
            preds.append(y_hat)
        # convert to numpy array
        preds = np.array(preds)
        preds = np.squeeze(preds)
        return preds
    
    def bptt(self,x,y,y_hat,z,t,layers,dLdh_next,dLdC_next,min_val=-10,max_val=10):
        
        # [1] dLdy
        dLdy = y - y_hat
        # [2] dLdh
        dLdh = dLdh_next + self.V.T @ dLdy 
        # [3] dLdC
        dLdC = dLdC_next + dLdh * layers[t]['o'] * (1-tanh(layers[t]['c'])**2)
        # [4] dLdC_til
        dLdC_til = dLdC * layers[t]['i']
        # [5] dLdo
        dLdo = dLdh * tanh(layers[t]['c'])
        # [6] dLdi
        dLdi = dLdC * layers[t]['c_til']
        # [7] dLdi
        dLdf = dLdC * layers[t]['c_prev']
        # [8] dLdz
        dLdz = self.W_f.T @ (layers[t]['f']*(1-layers[t]['f'])*dLdf) + \
        self.W_i.T @ (layers[t]['i']*(1-layers[t]['i'])*dLdi) + \
        self.W_c.T @ ((1-layers[t]['c_til']**2)*dLdC_til) + \
        self.W_o.T @ (layers[t]['o']*(1-layers[t]['o'])*dLdo)
        # update dLdC_next and dLdh_next
        dLdC_next = layers[t]['f'] * dLdC
        dLdh_next = dLdz[:self.hidden_dim]
        # calculate model parameter gradient
        dLdV = dLdy @ layers[t]['h'].T
        dLdb_y = dLdy
        dLdW_c = dLdC_til @ z.T
        dLdb_c = dLdC_til
        dLdW_o = dLdo @ z.T
        dLdb_o = dLdo
        dLdW_i = dLdi @ z.T
        dLdb_i = dLdi
        dLdW_f = dLdf @ z.T
        dLdb_f = dLdf
        
        return dLdC_next, dLdh_next,dLdV,dLdb_y,dLdW_c, dLdb_c, dLdW_o, dLdb_o, dLdW_i, dLdb_i, dLdW_f,dLdb_f
    
    def train (self,X,Y,epochs,learning_rate,min_val,max_val,verbose = True):
        # storage lost
        losses = []
        
        for epoch in range(epochs):
            # calculate loss
            loss = self.calc_loss(X,Y)
            losses.append(loss)
            title = f'epoch: {epoch} loss: {loss}'
            if verbose: print(title)

            layers = [] # layers storage params in forward pass
    
            # FORWARD PASS
            # init previous state
            h_prev = np.zeros((self.hidden_dim,1))
            C_prev = np.zeros((self.hidden_dim,1))

            for t in range(Y.shape[0]):
                x,y = X[t],Y[t]
                # feed forward
                h_prev,C_prev,f_t,i_t,o_t,c_til,c_t,h_t,y_hat = self.forward_pass(h_prev,C_prev,x)
                layers.append({'h_prev':h_prev,
                               'c_prev':C_prev,
                               'f':f_t,
                               'i':i_t,
                               'o':o_t,
                               'c_til':c_til,
                               'c':c_t,
                               'h':h_t,
                               'y_hat':y_hat}
                             )
                
            # BACKWARD PASS
            # init grads matrices
            # in the begining matrices
            dLdV = np.zeros(self.V.shape)
            dLdb_y = np.zeros(self.b_y.shape)
            dLdW_c = np.zeros(self.W_c.shape)
            dLdb_c = np.zeros(self.b_c.shape)
            dLdW_o = np.zeros(self.W_o.shape)
            dLdb_o = np.zeros(self.b_o.shape)
            dLdW_i = np.zeros(self.W_i.shape)
            dLdb_i = np.zeros(self.b_i.shape)
            dLdW_f = np.zeros(self.W_f.shape)
            dLdb_f = np.zeros(self.b_f.shape)

            dLdC_next = np.zeros((self.hidden_dim,1))
            dLdh_next = np.zeros((self.hidden_dim,1))
            
            for t in range(Y.shape[0]):
                # x,y
                x,y = X[t],Y[t]
                # z
                z = np.concatenate((layers[t]['h'],x))
                # y_hat
                y_hat = layers[t]['y_hat']
                # backward pass
                dLdC_next, dLdh_next,dLdV_t,dLdb_y_t,dLdW_c_t, dLdb_c_t, dLdW_o_t, dLdb_o_t, dLdW_i_t, dLdb_i_t, dLdW_f_t,dLdb_f_t = self.bptt(x,y,y_hat,z,t,layers,dLdh_next,dLdC_next,min_val= min_val,max_val=max_val)
                # accumulate gradients
                dLdV += dLdV_t
                dLdb_y += dLdb_y_t
                dLdW_c += dLdW_c_t
                dLdb_c += dLdb_c_t
                dLdW_o += dLdW_o_t
                dLdb_o += dLdb_o_t
                dLdW_i += dLdW_i_t
                dLdb_i += dLdb_i_t
                dLdW_f += dLdW_f_t
                dLdb_f += dLdb_f_t

            # gradient descent
            self.V += learning_rate * dLdV
            self.b_y += learning_rate * dLdb_y
            self.W_c += learning_rate * dLdW_c
            self.b_c += learning_rate * dLdb_c
            self.W_o += learning_rate * dLdW_o
            self.b_o += learning_rate * dLdb_o
            self.W_i += learning_rate * dLdW_i
            self.b_i += learning_rate * dLdb_i
            self.W_f += learning_rate * dLdW_f
            self.b_f += learning_rate * dLdb_f
        
        return losses