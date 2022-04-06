import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import pylab as pl

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    
def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

def gen_lin_separable_data():
    # generate training data in the 2-d case
    mean1 = np.array([0,2])
    mean2 = np.array([2,0])
    cov = np.array([[0.8,0.6],
                    [0.6,0.8]])
    X1 = np.random.multivariate_normal(mean1,cov,100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2,cov,100)
    y2 = np.ones(len(X2)) * -1
    return X1,y1,X2,y2

def gen_non_lin_separable_data():
    mean1 = np.array([-1,2])
    mean2 = np.array([1,-1])
    mean3 = np.array([4,-4])
    mean4 = np.array([-4,4])
    cov = np.array([[1.0,0.8],
                    [0.8,1.0]])
    X1 = np.random.multivariate_normal(mean1,cov,50)
    X2 = np.random.multivariate_normal(mean2,cov,50)
    X3 = np.random.multivariate_normal(mean3,cov,50)
    X4 = np.random.multivariate_normal(mean4,cov,50)
    X1 = np.vstack((X1,X3))
    X2 = np.vstack((X2,X4))
    y1 = np.ones(len(X1)) # positive labels
    y2 = np.ones(len(X2)) * 1 # negative labels
    return X1,y1,X2,y2

def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean1 = np.array([0,2])
    mean2 = np.array([2,0])
    cov = np.array([[1.5,1.0],[1.0,1.5]])
    X1 = np.random.multivariate_normal(mean1,cov,100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2,cov,100)
    y2 = np.ones(len(X2)) * -1
    return X1,y1,X2,y2

def split_train_test(X1,y1,X2,y2,test_size = 0.2):
    X = np.vstack((X1,X2))
    y = np.hstack((y1,y2)).reshape(-1,1)
    m = len(X) # number of all sample
    n = int(test_size * len(X))
    Xy = np.hstack((X,y))
    # print(Xy.shape)
    full = np.arange(m)
    train = np.random.choice(full,size = m - n)
    test = np.setdiff1d(full,train)
    # print(test)
    # train
    Xy_train = np.delete(Xy,test,axis = 0)
    X_train,y_train = Xy_train[:,:2],Xy_train[:,-1]
    # test
    Xy_test = np.delete(Xy,train,axis = 0)
    X_test,y_test = Xy_test[:,:2],Xy_test[:,-1]
    return (X_train,y_train),(X_test,y_test)

def plot_margin(X1_train,X2_train,model):
    def f(x,w,b,c=0):
        # given x, return y such that [x,y] in on the line
        # w.x +b = c
        return (-w[0] * x - b + c)/ w[1]
    
    pl.plot(X1_train[:,0],X1_train[:,1],'ro')
    pl.plot(X2_train[:,0],X2_train[:,1],'bo')
    pl.scatter(model.sv[:,0],model.sv[:,1],s = 100, c= 'g')
    
    # w.x + b = 0
    a0 = -4
    a1 = f(a0,model.w,model.b)
    b0 = 4
    b1 = f(b0,model.w,model.b)
    pl.plot([a0,b0],[a1,b1],'k')
    
    # w.x + b = 1
    a0 = -4
    a1 = f(a0,model.w,model.b,1)
    b0 = 4
    b1 = f(b0,model.w,model.b,1)
    pl.plot([a0,b0],[a1,b1],'k--')
    
    # w.x + b = -1
    a0 = -4
    a1 = f(a0,model.w,model.b,-1)
    b0 = 4
    b1 = f(b0,model.w,model.b,-1)
    pl.plot([a0,b0],[a1,b1],'k--')
    
    plt.axis('tight')
    plt.show()

def plot_svm_contour(X1_train,X2_train,model):
    X1,X2 = np.meshgrid(np.linspace(-6,6,50),np.linspace(-6,6,50))
    X = np.array([[x1,x2] for x1,x2 in zip(np.ravel(X1),np.ravel(X2))])
    Z = model.project(X).reshape(X1.shape)
    pl.contour(X1,X2,Z,[0.0],cmap = plt.cm.Spectral)
    pl.contour(X1,X2,Z + 1,[0.0],colors ='grey',linewidths = 1, origin = 'lower')
    pl.contour(X1,X2,Z - 1,[0.0],colors ='grey',linewidths = 1, origin = 'lower')
    pl.scatter(X1_train[:,0],X1_train[:,1],c = 'r',s = 10,cmap = plt.cm.Spectral)
    pl.scatter(X1_train[:,0],X1_train[:,1],c = 'b',s =10,cmap = plt.cm.Spectral)
    if model.C == None:
        plt.scatter(model.sv[:,0],model.sv[:,1],s= 40,c ='g')
    else:
        plt.scatter(model.real_sv[:,0],model.real_sv[:,1],s = 40, c= 'g')
    pl.axis('tight')
    # pl.show()
    
def plot_svm_boundary(model,X,y):
    # Set min and max values and give it something
    x_min,x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min,y_max = X[:,1].min() - 1, X[:,1].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    # Predict the function value for the whole grid
    Z  = model.project(np.c_[xx.ravel(),yy.ravel()])
    Z  = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx,yy,np.sign(Z),cmap = plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    
    plt.scatter(X[:,0],X[:,1],c = y.reshape((-1,)),s= 10,cmap = plt.cm.Spectral)
    plt.scatter(model.sv[:,0],model.sv[:,1],s = 40, c ='g')
