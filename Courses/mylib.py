# 3 layer network 
# 1
def activation_sigmoid(z):
    return np.power((1 + np.exp(-z)) ,-1)
######################################################################
# 2
def activation_tanh(z):
    return np.tanh(z)
######################################################################
# 3
def compute_cost(a,Y):
    cost = -(Y*np.log(a) + (1 - Y)*np.log(1-a))
    cost = np.mean(cost)
    return cost
######################################################################
# 4
def layer_sizes(X,Y,n_h = 4):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer
    m   = X.shape[1]
    return (n_x,n_h,n_y,m)
######################################################################
# 5
def initialize_with_parameters(n_x,n_h,n_y,seed =1):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(seed) # we set up a seed so that your output 
    #matches ours although the initialization is random.
    
    W1 = np.random.randn(n_h,n_x) *0.01 
    b1 = np.random.randn(n_h,1)   *0.01
    W2 = np.random.randn(n_y,n_h) *0.01
    b2 = np.random.randn(n_y,1)   *0.01
    #If condition is true = > next
    
    assert (W1.shape == (n_h,n_x))
    assert (b1.shape == (n_h,1))
    assert (W2.shape == (n_y,n_h))
    assert (b2.shape == (n_y,1))
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    
    return params
######################################################################
# 6
def forward_propagation(params,X):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters 
    (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)    
    Z1 = W1 @ X  + b1
    A1 = activation_tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = activation_sigmoid(Z2) 
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    assert(A2.shape == (1,X.shape[1]))
    return A2,cache
######################################################################
# 7
def backward_propagation(params,cache,X,Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with 
    respect to different parameters
    """
    n_x,n_h,n_y,m = layer_sizes(X,Y)
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = params["W1"]
    W2 = params["W2"]
    
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2
    
    dZ2 = A2 - Y 
    dW2 = 1/m * (dZ2 @ A1.T)
    db2 = 1/m * np.sum(dZ2,axis =1,keepdims = True)
    dZ1 = np.matmul(W2.T,dZ2) * (1 - np.power(A1,2))
    dW1 = 1/m * (dZ1 @ X.T)
    db1 = 1/m * np.sum(dZ1,axis =1,keepdims = True)
    
    grads = {"dW2": dW2,
             "db2": db2,
             "dW1": dW1,
             "db1": db1}
    
    return grads
######################################################################
# 8
def propagate(params,X,Y):
    
    A2,cache = forward_propagation(params,X)
    
    grads = backward_propagation(params,cache,X,Y)
    
    cost = compute_cost(A2,Y)
    
    return grads, cost
######################################################################
# 9
def update_parameters(params,grads,learning_rate):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter gradient decent
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    
    return params
######################################################################
# 10
def optimize(params,X,Y,num_iterations,learning_rate,print_cost = False):
    
    costs = []
    
    for i in range(0,num_iterations):
        
        grads,cost = propagate(params,X,Y)
        
        params = update_parameters(params,grads,learning_rate)
        
        if i % 500 == 0:
            costs.append(cost)
            if print_cost :
                print("Cost after iteration %i: %f" %(i,cost))
    
    return params,grads,costs
######################################################################
# 11
def predict(params,X,accept_rate =0.5):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    A2,cache = forward_propagation(params,X) 
    
    Y_prediction = np.zeros(A2.shape)
    
    #Y_prediction = (A2 > accept_rate)
    
    for i in range (A2.shape[1]):
        if A2[0,i] > accept_rate:
            Y_prediction[0,i] =1
        else:
            Y_prediction[0,i] =0
    
    #assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

######################################################################
# 12
def process_image(my_image,base_size = (64,64)):
    my_image = my_image.resize(base_size,Image.BICUBIC)   
    plt.imshow(my_image)
    X_new = np.array(my_image)
    X_new = X_new.ravel() 
    X_new = X_new.reshape(X_new.shape[0],1)    
    return X_new
######################################################################
# 13
def calc_train_error(X_train,Y_train,params):
    Y_train_prediction = predict(params,X_train,accept_rate =0.5)
    mse_train = np.mean((Y_train - Y_train_prediction)**2)
    rmse_train = np.sqrt(mse_train) * 100
    return Y_train_prediction,rmse_train
######################################################################
# 14
def calc_test_error(X_test,Y_test,params):
    Y_test_prediction = predict(params,X_test,accept_rate =0.5)
    mse_test = np.mean((Y_test - Y_test_prediction)**2)
    rmse_test = np.sqrt(mse_test) * 100
    return Y_test_prediction,rmse_test
######################################################################
# 15
def calc_metrics(X_train,Y_train,X_test,Y_test,params,train_accept,valid_accept,diagnostic):
    Y_train_prediction,train_error = calc_train_error(X_train,Y_train,params)
    Y_test_prediction,valid_error = calc_test_error(X_test,Y_test,params)
    
    if diagnostic:
        if valid_error > valid_accept:
            if train_error > train_accept:
                print("High Variance and High Bias" + \
                      "\n" + "More Data and regulazation for high variance" +
                      "\n" + "Bigger Network")
            else:  print("High Variance " + \
                  "\n" + "More Data and regulazation for high variance")
        else:
            if train_error > train_accept:
                print("High Bias" + "\n" + "Bigger Network")
            else:  print("Low Variance and Low Bias => O.K ")
                
    return Y_train_prediction,train_error,Y_test_prediction,valid_error
######################################################################
# 16
def n_model(X_train,Y_train,X_dev,Y_dev,n_h,num_iterations = 1000,learning_rate = 0.5,\
            print_cost = False,train_accept = 10,valid_accept = 15,diagnostic = False):

    n_x,n_h,n_y,m = layer_sizes(X_train,Y_train,n_h)
    
    params = initialize_with_parameters(n_x,n_h,n_y,seed = 1)
    
    params, grads, costs = optimize(params,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    Y_train_prediction,train_error,Y_test_prediction,valid_error = calc_metrics\
    (X_train,Y_train,X_dev,Y_dev,params,train_accept,valid_accept,diagnostic)
    
    print("train_error: " +str(train_error))
    
    print("valid_error: " +str(valid_error))
    
    d = {"costs"              : costs,
         "train_error"        : train_error,
         "valid_error"        : valid_error,
         "Y_train_prediction" : Y_train_prediction,
         "Y_test_prediction"  : Y_test_prediction,
         "parameters"         : params,
         "learning_rate"      : learning_rate,
         "num_interations"    : num_iterations
          }
        
    return d
######################################################################
# 17
def layer_early_stop(X_train,Y_train,X_dev,Y_dev,hidden_layer_sizes,num_iteration = 3000\
                     ,learning_rate = 0.01,print_cost = True,train_accept = 10,valid_accept = 15,diagnostic = True):
    valid_now = 0
    valid_old = 0 
    d_now = {}
    
    for i,n_h in enumerate(hidden_layer_sizes):        
        d_old = d_now
        d_now = n_model(X_train,Y_train,X_dev,Y_dev,n_h,num_iteration,learning_rate,print_cost\
                        ,train_accept,valid_accept,diagnostic)
        valid_old = valid_now
        valid_now = d_now["valid_error"]
        print("valid_old:" + str(valid_old))
        print("valid_now:" + str(valid_now))
        if (valid_now > valid_old) & (i!=0):
            print("Early stopping, Now!")
            return d_old
            break
    return d_now
######################################################################
# 18
def iteration_early_stop(X_train,Y_train,X_dev,Y_dev,n_h,num_iter,learning_rate = 0.01,\
                         print_cost = True,train_accept = 10,valid_accept = 15,diagnostic = True):
    valid_now = 0
    valid_old = 0 
    d_now = {}
    
    for i,num_iteration in enumerate(num_iter):        
        d_old = d_now
        d_now = n_model(X_train,Y_train,X_dev,Y_dev,n_h,num_iteration,learning_rate,\
                        print_cost,train_accept,valid_accept,diagnostic)
        valid_old = valid_now
        valid_now = d_now["valid_error"]
        print("valid_old:" + str(valid_old))
        print("valid_now:" + str(valid_now))
        if (valid_now > valid_old) & (i!=0):
            print("Early stopping, Now!")
            return d_old
            break
    return d_now
######################################################################
# 19
# L2
def wdecay_compute_cost(a,Y,W1,W2,lambd):
    m = Y.shape[1]
    cost = -(Y * np.log(a) + (1 - Y) * np.log(1-a))
    cost = np.mean(cost)
    cost_w = cost + lambd/(2*m) * np.linalg.norm(W1)**2 + lambd/(2*m) * np.linalg.norm(W2)**2
    return cost,cost_w
######################################################################
# 20
def wdecay_model(X_train,Y_train,X_dev,Y_dev,n_h,num_iterations = 10000\
                 ,learning_rate = 0.5,print_cost = False,lambd = 0.1,train_accept = 10,\
                 valid_accept = 15,diagnostic = True):

    n_x,n_h,n_y,m = layer_sizes(X_train,Y_train,n_h)
    
    params = initialize_with_parameters(n_x,n_h,n_y,seed = 1)
    
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    
    #params, grads, costs,costs_w = wdecay_optimize(params,X_train,Y_train,num_iterations,learning_rate,lambd,print_cost)    
    
    costs   = []
    costs_w = []
    
    for i in range(0,num_iterations):
        
    #grads,cost,cost_w = wdecay_propagate(params,X,Y,lambd)
        
        #A2,cache = forward_propagation(params,X)
        
        Z1 = W1 @ X_train  + b1
        A1 = activation_tanh(Z1)
        Z2 = W2 @ A1 + b2
        A2 = activation_sigmoid(Z2) 
        
        #grads = wdecay_backward_propagation(params,cache,X,Y,lambd)
        
        dZ2 = A2 - Y_train        
        dW2 = 1/m * (dZ2 @ A1.T)
        dLdW2 = dW2 + lambd/m * W2
        db2 = 1/m * np.sum(dZ2,axis =1,keepdims = True)
        dZ1 = np.matmul(W2.T,dZ2) * (1 - np.power(A1,2))
        dW1 = 1/m * (dZ1 @ X_train.T)
        dLdW1 = dW1 + lambd/m * W1
        db1 = 1/m * np.sum(dZ1,axis =1,keepdims = True)
        
        #params     = wdecay_update_parameters(params,grads,learning_rate,lambd,m)
        
        W1 = W1 - learning_rate *dLdW1
        #W1 = W1*(1-lambd*learning_rate) - learning_rate*dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate *dLdW2
        #W2 = W2*(1-lambd*learning_rate) - learning_rate*dW2
        b2 = b2 - learning_rate * db2
        
        cost,cost_w =  wdecay_compute_cost(A2,Y_train,W1,W2,lambd)
        
        if i % 500 == 0:
            costs.append(cost)
            costs_w.append(cost_w)
            if print_cost :
                print("Cost after iteration %i: %f" %(i,cost))
                print("Cost_w after iteration %i: %f" %(i,cost_w))
                
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    
    Y_train_prediction,train_error,Y_test_prediction,valid_error = \
    calc_metrics(X_train,Y_train,X_dev,Y_dev,params,train_accept,valid_accept,diagnostic)
    

    print("train_error: " +str(train_error))
    
    print("valid_error: " +str(valid_error))
    
    #print("dW1:" +str(dW1.shape))
    #print("dLdW1:" +str(dLdW1.shape))
    #print("dW2:" +str(dW2.shape))
    #print("dLdW2:" +str(dLdW2.shape))
    
    d = {"costs"              : costs,
         "costs_w"            : costs_w,
         "train_error"        : train_error,
         "valid_error"        : valid_error,
         "Y_train_prediction" : Y_train_prediction,
         "Y_test_prediction"  : Y_test_prediction,
         "parameters"         : params,
         "learning_rate"      : learning_rate,
         "num_interations"    : num_iterations
          }
        
    return d
######################################################################
# 21
def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape
  #step1: calculate mean
    mu = 1./N * np.sum(x, axis = 0)
  #step2: subtract mean vector of every trainings example
    xmu = x - mu
  #step3: following the lower branch - calculation denominator
    sq = xmu ** 2
  #step4: calculate variance
    var = 1./N * np.sum(sq, axis = 0)
  #step5: add eps for numerical stability, then sqrt
    sqrtvar = np.sqrt(var + eps)
  #step6: invert sqrtwar
    ivar = 1./sqrtvar
  #step7: execute normalization
    xhat = xmu * ivar # (1,N) * (D,N) each feature multiply with each sqrt 
  #step8: Nor the two transformation steps
    gammax = gamma * xhat
  #step9
    out = gammax + beta
  #store intermediate
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    return out, cache
######################################################################
# 22
def batchnorm_backward(dout, cache):
  #unfold the variables stored in cache
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
  #get the dimensions of the input/output
    N,D = dout.shape
  #step9
    dbeta = np.sum(dout, axis=0)
    dgammax = dout #not necessary, but more understandable
  #step8
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma
  #step7
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar
  #step6
    dsqrtvar = -1. /(sqrtvar**2) * divar
  #step5
    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
  #step4
    dsq = 1. /N * np.ones((N,D)) * dvar
  #step3
    dxmu2 = 2 * xmu * dsq
  #step2
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
  #step1
    dx2 = 1. /N * np.ones((N,D)) * dmu
  #step0
    dx = dx1 + dx2
    return dx, dgamma, dbeta
######################################################################
# 23
# dropout function
def dropout(A,keepdrop):
    assert keepdrop <1
    D = np.random.rand(A.shape[0],A.shape[1]) < keepdrop
    A_dropout = A * D
    return A_dropout

######################################################################
# 24

def layer_sizes_n2(X,Y,n_h1=5,n_h2=3):
    """
    Argument:
    n_x  -- size of the input layer
    n_h1 -- size of the hidden layer1
    n_h2 -- size of the hidden layer2
    n_y  -- size of the output layer
    m    -- sample number
    """
    n_x = X.shape[0] # size of input layer
    n_y = Y.shape[0] # size of output layer
    m   = X.shape[1]
    return (n_x,n_h1,n_h2,n_y,m)

######################################################################
# 25

def initialize_with_parameters_n2(n_x, n_h1, n_h2, n_y,seed):
    """
    Argument:
    n_x  -- size of the input layer
    n_h1 -- size of the hidden layer1
    n_h2 -- size of the hidden layer1
    n_y  -- size of the output layer
    seed -- seed of random
    
    X    -- (n_x,1) or (n_x,samples)
    
    params = 
    W1   -- (n_h1, n_x)
    b1   -- (n_h1, 1)
    W2   -- (n_h2, n_h1)
    b2   -- (n_h2, 1)
    W3   -- (n_y, n_h2)
    b3   -- (n_y, 1)
    
    Y    -- (n_y,1)
    """
    np.random.seed(seed) 
    
    W1 = np.random.randn(n_h1,n_x)  *0.01 
    b1 = np.random.randn(n_h1,1)    *0.01
    W2 = np.random.randn(n_h2,n_h1) *0.01
    b2 = np.random.randn(n_h2,1)    *0.01
    W3 = np.random.randn(n_y,n_h2)  *0.01
    b3 = np.random.randn(n_y,1)     *0.01 
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    return params

######################################################################
# 26

def activation_relu(z):
    z_out = []
    for i in z.ravel():
        if i > 0:
            z_out.append(i)
        else: z_out.append(0)
    z_out = np.reshape(z_out,z.shape)
    return np.array(z_out)

######################################################################
# 26

def forward_propagation_n2(params,X):

    # Retrieve each parameter from the dictionary "parameters"
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)      
    # LINEAR -> TANH -> LINEAR -> TANH -> LINEAR -> SIGMOID
    
    Z1 = W1 @ X  + b1           # LINEAR
    A1 = activation_tanh(Z1)    # RELU
    Z2 = W2 @ A1 + b2           # LINEAR
    A2 = activation_tanh(Z2)    # RELU  
    Z3 = W3 @ A2 + b3           # LINEAR
    A3 = activation_sigmoid(Z3) # SIGMOID
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3}
    #assert(A3.shape == (1,X.shape[1]))
    return A3,cache

######################################################################
# 27

def backward_propagation_n2(params,cache,X,Y):
    
    m = X.shape[1]    
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]
    
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2
 
    dLdZ3 = A3 - Y
    #dLdZ3 = dLdA3 *(1 - np.power(A3,2))
    dLdW3 = 1/m  * dLdZ3 @ A2.T                        #dLdW3 = 1/m  * np.dot(dLdZ3, A2.T) 
    dLdb3 = 1/m  * np.sum(dLdZ3, axis=1, keepdims = True)
    
    dLdZ2 = (dLdZ3.T @ W3).T * (1 - np.power(A2,2))  #A2*(1-A2)   
    
    dLdW2 = 1/m * dLdZ2 @ A1.T                        #dLdW2 = 1./m * np.dot(dLdZ2, A1.T)
    dLdb2 = 1/m * np.sum(dLdZ2, axis=1, keepdims = True)

    dLdZ1 = (dLdZ2.T @ W2).T * (1 - np.power(A1,2))  # A1*(1-A1) 
    
    dLdW1 = 1/m * dLdZ1 @ X.T                         #dLdW1 = 1./m * np.dot(dLdZ1, X.T)  
    dLdb1 = 1/m * np.sum(dLdZ1, axis=1, keepdims = True)
    
    grads = {"dLdW3": dLdW3,
             "dLdb3": dLdb3,
             "dLdW2": dLdW2,
             "dLdb2": dLdb2,
             "dLdW1": dLdW1,
             "dLdb1": dLdb1}   


    
    return grads

 ######################################################################
# 28

def propagate_n2(params,X,Y):
    
    A3,cache = forward_propagation_n2(params,X)
    
    grads = backward_propagation_n2(params,cache,X,Y)

    cost = compute_cost(A3,Y)
    
    return grads, cost

######################################################################
# 29

def update_parameters_n2(params,grads,learning_rate):

    # Retrieve each parameter from the dictionary "parameters"
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    
    # Retrieve each gradient from the dictionary "grads"
    dLdW1 = grads["dLdW1"]
    dLdb1 = grads["dLdb1"]
    dLdW2 = grads["dLdW2"]
    dLdb2 = grads["dLdb2"]
    dLdW3 = grads["dLdW3"]
    dLdb3 = grads["dLdb3"]
    
    # Update rule for each parameter gradient decent
    W1 = W1 - learning_rate * dLdW1
    b1 = b1 - learning_rate * dLdb1
    W2 = W2 - learning_rate * dLdW2
    b2 = b2 - learning_rate * dLdb2
    W3 = W3 - learning_rate * dLdW3
    b3 = b3 - learning_rate * dLdb3
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    return params

def optimize_n2(params,X,Y,num_iterations,learning_rate,print_cost = False):
    
    costs = []
    
    for i in range(0,num_iterations):
        
        grads,cost = propagate_n2(params,X,Y)
        
        params = update_parameters_n2(params,grads,learning_rate)
        
        if i % 500 == 0:
            costs.append(cost)
            if print_cost :
                print("Cost after iteration %i: %f" %(i,cost))
    
    return params,grads,costs

 ######################################################################
# 30

def predict_n2(params,X,accept_rate =0.5):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    A_out,cache = forward_propagation_n2(params,X) 
    
    Y_prediction = np.zeros(A_out.shape)
    
    #Y_prediction = (A_out > accept_rate)
    
    for i in range (A_out.shape[1]):
        if A_out[0,i] > accept_rate:
            Y_prediction[0,i] =1
        else:
            Y_prediction[0,i] =0
    
    #assert(Y_prediction.shape == (1,m))
    
    return Y_prediction

 ######################################################################
# 31

def calc_train_error_n2(X_train,Y_train,params):
    Y_train_prediction = predict_n2(params,X_train,accept_rate =0.5)
    mse_train = np.mean((Y_train - Y_train_prediction)**2)
    rmse_train = np.sqrt(mse_train) * 100
    return Y_train_prediction,rmse_train

 ######################################################################
# 32

def calc_test_error_n2(X_test,Y_test,params):
    Y_test_prediction = predict_n2(params,X_test,accept_rate =0.5)
    mse_test = np.mean((Y_test - Y_test_prediction)**2)
    rmse_test = np.sqrt(mse_test) * 100
    return Y_test_prediction,rmse_test

 ######################################################################
# 33
def calc_metrics_n2(X_train,Y_train,X_test,Y_test,params,train_accept,valid_accept,diagnostic):
    Y_train_prediction,train_error = calc_train_error_n2(X_train,Y_train,params)
    Y_test_prediction,valid_error = calc_test_error_n2(X_test,Y_test,params)
    
    if diagnostic:
        if valid_error > valid_accept:
            if train_error > train_accept:
                print("High Variance and High Bias" + \
                      "\n" + "More Data and regulazation for high variance" +
                      "\n" + "Bigger Network")
            else:  print("High Variance " + \
                  "\n" + "More Data and regulazation for high variance")
        else:
            if train_error > train_accept:
                print("High Bias" + "\n" + "Bigger Network")
            else:  print("Low Variance and Low Bias => O.K ")
                
    return Y_train_prediction,train_error,Y_test_prediction,valid_error


 ######################################################################
# 34

def n2_modela(X_train,Y_train,X_dev,Y_dev,n_h1 = 10,n_h2 = 10,num_iterations = 1000,learning_rate = 0.5,\
            print_cost = False,train_accept = 10,valid_accept = 15,diagnostic = False):

    n_x,n_h1,n_h2,n_y,m = layer_sizes_n2(X_train,Y_train,n_h1,n_h2) ##################################
    
    params = initialize_with_parameters_n2(n_x, n_h1, n_h2, n_y,seed =1) #############################
    
    costs = []
    
    for i in range(0,num_iterations):
        
        A3,cache = forward_propagation_n2(params,X_train)
    
        grads = backward_propagation_n2(params,cache,X_train,Y_train)

        cost = compute_cost(A3,Y_train)
        
        params = update_parameters_n2(params,grads,learning_rate)
        
        if i % 500 == 0:
            costs.append(cost)
            if print_cost :
                print("Cost after iteration %i: %f" %(i,cost))
    
    Y_train_prediction,train_error,Y_test_prediction,valid_error = calc_metrics_n2\
    (X_train,Y_train,X_dev,Y_dev,params,train_accept,valid_accept,diagnostic)
    
    print("train_error: " +str(train_error))
    
    print("valid_error: " +str(valid_error))
    
    d = {"costs"              : costs,
         "train_error"        : train_error,
         "valid_error"        : valid_error,
         "Y_train_prediction" : Y_train_prediction,
         "Y_test_prediction"  : Y_test_prediction,
         "parameters"         : params,
         "learning_rate"      : learning_rate,
         "num_interations"    : num_iterations
          }
        
    return d

 ######################################################################
# 35
def n2_modelb(X_train,Y_train,X_dev,Y_dev,n_h1 = 10,n_h2 = 10,num_iterations = 1000,learning_rate = 0.5,\
            print_cost = False,train_accept = 10,valid_accept = 15,diagnostic = False):

    n_x,n_h1,n_h2,n_y,m = layer_sizes_n2(X_train,Y_train,n_h1,n_h2) ##################################
    
    params = initialize_with_parameters_n2(n_x, n_h1, n_h2, n_y,seed =1) #############################
    
    params,grads,costs = optimize_n2(params,X,Y,num_iterations,learning_rate,print_cost)
    
    Y_train_prediction,train_error,Y_test_prediction,valid_error = calc_metrics_n2\
    (X_train,Y_train,X_dev,Y_dev,params,train_accept,valid_accept,diagnostic)
    
    print("train_error: " +str(train_error))
    
    print("valid_error: " +str(valid_error))
    
    d = {"costs"              : costs,
         "train_error"        : train_error,
         "valid_error"        : valid_error,
         "Y_train_prediction" : Y_train_prediction,
         "Y_test_prediction"  : Y_test_prediction,
         "parameters"         : params,
         "learning_rate"      : learning_rate,
         "num_interations"    : num_iterations
          }
        
    return d

######################################################################
# 36

def batchnorm_forward_w(x, gamma, beta, eps):
    N, D = x.shape
    mu = 1./N * np.sum(x, axis = 0)
    xmu = x - mu
    sq = xmu ** 2
    var = 1./N * np.sum(sq, axis = 0)
    sqrtvar = np.sqrt(var + eps)
    ivar = 1./sqrtvar
    xhat = xmu * ivar
    gammax = gamma * xhat
    out = gammax + beta
    cache = (xhat,gamma,xmu,ivar,sqrtvar,var,eps)
    return out, cache

######################################################################
# 37

def batchnorm_backward_w(dout, cache):
    xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
    N,D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgammax = dout 
    dgamma = np.sum(dgammax*xhat, axis=0)
    dxhat = dgammax * gamma
    divar = np.sum(dxhat*xmu, axis=0)
    dxmu1 = dxhat * ivar
    dsqrtvar = -1. /(sqrtvar**2) * divar
    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
    dsq = 1. /N * np.ones((N,D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
    dx2 = 1. /N * np.ones((N,D)) * dmu
    dx = dx1 + dx2
    cache_bb = {"dx": dx,
                "dgamma": dgamma,
                "dbeta": dbeta}
    return cache_bb

######################################################################
# 38

def forward_propagation_n2w(params,X):

    # Retrieve each parameter from the dictionary "parameters"
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    gamma1 = params["gamma1"]
    beta1 = params["beta1"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)      
    # LINEAR -> TANH -> LINEAR -> TANH -> LINEAR -> SIGMOID
    
    Z1 = W1 @ X  + b1           # LINEAR
    
    A11 = activation_tanh(Z1)    # RELU
    
    A1,cache1 = batchnorm_forward(A11, gamma = gamma1, beta =beta1, eps = 0.001)
    
    Z2 = W2 @ A1 + b2           # LINEAR
    
    A2 = activation_tanh(Z2)    # RELU  
    
    Z3 = W3 @ A2 + b3           # LINEAR
    
    A3 = activation_sigmoid(Z3) # SIGMOID
    
    cache = {"Z1": Z1,
             "A1": A11,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3,
             "cache1": cache1}
    #assert(A3.shape == (1,X.shape[1]))
    return A3,cache

######################################################################
# 39

def initialize_with_parameters_n2w(n_x, n_h1, n_h2, n_y,seed):

    np.random.seed(seed) 
    
    W1 = np.random.randn(n_h1,n_x)  *0.01 
    b1 = np.random.randn(n_h1,1)    *0.01
    W2 = np.random.randn(n_h2,n_h1) *0.01
    b2 = np.random.randn(n_h2,1)    *0.01
    W3 = np.random.randn(n_y,n_h2)  *0.01
    b3 = np.random.randn(n_y,1)     *0.01 
    #gamma1 = np.random.randn(1)     *0.01
    gamma1 = 1
    #beta1 = np.random.randn(1)      *0.01
    beta1 = 1
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
             "gamma1": gamma1,
             "beta1": beta1}
    
    return params

 ######################################################################
# 40

def forward_propagation_n2w(params,X):

    # Retrieve each parameter from the dictionary "parameters"
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    gamma1 = params["gamma1"]
    beta1 = params["beta1"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)      
    # LINEAR -> TANH -> LINEAR -> TANH -> LINEAR -> SIGMOID
    
    Z1 = W1 @ X  + b1           # LINEAR
    
    A11 = activation_tanh(Z1)    # RELU
    
    A1,cache1 = batchnorm_forward(A11, gamma = gamma1, beta =beta1, eps = 0.001)
    
    Z2 = W2 @ A1 + b2           # LINEAR
    
    A2 = activation_tanh(Z2)    # RELU  
    
    Z3 = W3 @ A2 + b3           # LINEAR
    
    A3 = activation_sigmoid(Z3) # SIGMOID
    
    cache = {"Z1": Z1,
             "A1": A11,
             "Z2": Z2,
             "A2": A2,
             "Z3": Z3,
             "A3": A3,
             "cache1": cache1}
    #assert(A3.shape == (1,X.shape[1]))
    return A3,cache

  ######################################################################
# 41

def backward_propagation_n2w(params,cache,X,Y):
    
    m = X.shape[1]    
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = params["W1"]
    W2 = params["W2"]
    W3 = params["W3"]
    
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    A3 = cache["A3"]
    
    cache1 = cache["cache1"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2
 
    dLdZ3 = A3 - Y
    #dLdZ3 = dLdA3 *(1 - np.power(A3,2))
    
    dLdW3 = 1./m  * np.dot(dLdZ3, A2.T) 
    dLdb3 = 1./m  * np.sum(dLdZ3, axis=1, keepdims = True)
    
    cache_bb = batchnorm_backward(A1, cache1)
    
    dLdZ2 = (dLdZ3.T @ W3).T * (1 - np.power(A2,2))  #A2*(1-A2)
    
    dLdW2 = 1./m * np.dot(dLdZ2, A1.T) 
    dLdb2 = 1./m * np.sum(dLdZ2, axis=1, keepdims = True)

    dLdZ1 = (dLdZ2.T @ W2).T * (1 - np.power(A1,2)) # A1*(1-A1) 
    
    dLdW1 = 1./m * np.dot(dLdZ1, X.T)
    dLdb1 = 1./m * np.sum(dLdZ1, axis=1, keepdims = True)
    
    grads = {"dLdW3": dLdW3,
             "dLdb3": dLdb3,
             "dLdW2": dLdW2,
             "dLdb2": dLdb2,
             "dLdW1": dLdW1,
             "dLdb1": dLdb1,
            "cache_bb": cache_bb
            }   
    
    return grads

   ######################################################################
# 42

def update_parameters_n2w(params,grads,learning_rate,learning_rate_w):

    # Retrieve each parameter from the dictionary "parameters"
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    W3 = params["W3"]
    b3 = params["b3"]
    
    gamma1 = params["gamma1"]
    beta1 = params["gamma1"]
    cache_bb = grads["cache_bb"]
    dgamma = cache_bb["dgamma"]
    dbeta = cache_bb["dbeta"]
    gamma1 = gamma1 - learning_rate_w * dgamma
    beta1 =  beta1  - learning_rate_w  * dbeta
    # Retrieve each gradient from the dictionary "grads"
    dLdW1 = grads["dLdW1"]
    dLdb1 = grads["dLdb1"]
    dLdW2 = grads["dLdW2"]
    dLdb2 = grads["dLdb2"]
    dLdW3 = grads["dLdW3"]
    dLdb3 = grads["dLdb3"]
    
    # Update rule for each parameter gradient decent
    W1 = W1 - learning_rate * dLdW1
    b1 = b1 - learning_rate * dLdb1
    W2 = W2 - learning_rate * dLdW2
    b2 = b2 - learning_rate * dLdb2
    W3 = W3 - learning_rate * dLdW3
    b3 = b3 - learning_rate * dLdb3
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3,
             "gamma1": gamma1,
             "beta1": beta1}
    
    return params

   ######################################################################
# 43

def n2w_model(X_train,Y_train,X_dev,Y_dev,n_h1 = 10,n_h2 = 10,num_iterations = 1000,learning_rate = 0.5,\
            learning_rate_w = 0.01, print_cost = False,train_accept = 10,valid_accept = 15,diagnostic = False):

    n_x,n_h1,n_h2,n_y,m = layer_sizes_n2(X_train,Y_train,n_h1,n_h2) ##################################
    
    params = initialize_with_parameters_n2w(n_x, n_h1, n_h2, n_y,seed =1) #############################
    
    costs = []
    
    for i in range(0,num_iterations):
        
        A3,cache = forward_propagation_n2w(params,X_train)
    
        grads = backward_propagation_n2w(params,cache,X_train,Y_train)

        cost = compute_cost(A3,Y_train)
        
        params = update_parameters_n2w(params,grads,learning_rate,learning_rate_w)
        
        if i % 500 == 0:
            costs.append(cost)
            if print_cost :
                print("Cost after iteration %i: %f" %(i,cost))
    
    Y_train_prediction,train_error,Y_test_prediction,valid_error = calc_metrics_n2\
    (X_train,Y_train,X_dev,Y_dev,params,train_accept,valid_accept,diagnostic)
    
    print("train_error: " +str(train_error))
    
    print("valid_error: " +str(valid_error))
    
    d = {"costs"              : costs,
         "train_error"        : train_error,
         "valid_error"        : valid_error,
         "Y_train_prediction" : Y_train_prediction,
         "Y_test_prediction"  : Y_test_prediction,
         "parameters"         : params,
         "learning_rate"      : learning_rate,
         "num_interations"    : num_iterations
          }
        
    return d


    
######################################################################
##################### ONLY FOR REFERENCES ############################
######################################################################


def backward_propagation_with_regularisation(X, Y, cache, lambd):
    """
    Implements the backward propagation of our baseline model to which we added an L2 regularisation.

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation()
    lambd -- regularisation hyperparameter, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y

    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd*W3/m
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd*W2/m
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd*W1/m
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """

    np.random.seed(1)

    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    D1 = np.random.rand(A1.shape[0],A1.shape[1])      # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = (D1<keep_prob)                               # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1*D1                                        # Step 3: shut down some neurons of A1
    A1 = A1/keep_prob                                 # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    D2 = np.random.rand(A2.shape[0],A2.shape[1])      # Step 1: initialize matrix D2 = np.random.rand(..., ...)
    D2 = (D2<keep_prob)                               # Step 2: convert entries of D2 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2*D2                                        # Step 3: shut down some neurons of A2
    A2 = A2/keep_prob                                 # Step 4: scale the value of neurons that haven't been shut down

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache

def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    Implements the backward propagation of our baseline model to which we added dropout.

    Arguments:
    X -- input dataset, of shape (2, number of examples)
    Y -- "true" labels vector, of shape (output size, number of examples)
    cache -- cache output from forward_propagation_with_dropout()
    keep_prob - probability of keeping a neuron active during drop-out, scalar

    Returns:
    gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
    """

    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = dA2*D2            # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2/keep_prob     # Step 2: Scale the value of neurons that haven't been shut down

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)

    dA1 = dA1*D1            # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1/keep_prob     # Step 2: Scale the value of neurons that haven't been shut down

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

