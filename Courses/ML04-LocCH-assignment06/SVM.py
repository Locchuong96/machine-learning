import numpy as np  
import sklearn  
import cvxopt

# Create a subclass SVM model
class MySVM(object):
    def __init__(self,kernel = 'linear', C = None, gamma = 1, dergree = 3, coef0 = 1,epsilon = 1e-5):
        self.kernel = kernel # kernel types (linear,polynomial,rbf,gaussian)
        self.C = C
        if self.C is not None:
            self.C  = float(self.C)
        self.gamma = gamma # for rbf_kernel of sklearn.metrics.pairwise
        self.dergree = 3 # for polynomial_kernel of sklearn.metrics.pairwise
        self.coef0 = 1 # for polynomial_kernel of sklearn.metrics.pairwise
        self.epsilon = epsilon # Alphas accepted rate
    def fit(self,X,y):
        '''
        Train the model to find w,b
        
        Arguments:
            X --- features, shape (n_samples,n_features)
            y --- binary labels, shape (n_samples,1)
        '''
        n_samples,n_features = X.shape
        # Gram matrix
        if self.kernel == 'linear':
            K = X.dot(X.T)
        elif self.kernel == 'polynomial':
            K = sklearn.metrics.pairwise.polynomial_kernel(X,X,degree = self.degree,gamma = self.gamma, coef0 = self.coef0)
        elif self.kernel == 'rbf' or self.kernel == 'gaussian':
            K = sklearn.metrics.pairwise.rbf_kernel(X,X,gamma =self.gamma)
        else:
            assert "Kernel %s is not supported!"
        
        P = cvxopt.matrix(np.multiply(np.outer(y, y), K))
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y.astype('float'), (1,n_samples))
        b = cvxopt.matrix(0.0)
        
        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples)*-1)) # -a_i <=0
            h = cvxopt.matrix(np.zeros(n_samples))  
        else:
            tmp1 = np.diag(np.ones(n_samples)*-1) # -a_i <=0
            tmp2 = np.identity(n_samples) # a_i < C
            G = cvxopt.matrix(np.vstack((tmp1,tmp2)),(n_samples *2,n_samples))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1,tmp2)),(n_samples * 2,1))
        # find the solution
        solution = cvxopt.solvers.qp(P,q,G,h,A,b)
        alphas= np.array(solution['x'])
        alphas_sv = (alphas > self.epsilon).reshape((-1,)) # a list of True False
        ind = np.arange(len(alphas)).reshape(-1,1)[alphas_sv] # find the index of alpha_i > epsilon
        a = alphas[alphas_sv].reshape((-1,)) # alphas > 0
        sv =X[alphas_sv] # support vectors
        sv_y = y[alphas_sv].reshape((-1,)) # y(i) s.t X[i] is support vector
        
        # Intecept
        # SVM with non-linear kernel
        b = 0
        count = 0
        real_sv = []
        for i in range(len(a)):
            if self.C != None and a[i] > self.C + self.epsilon: # violate the condition a_i <= C
                continue
            else: # if C is non meaning non soft margin or a_i <= C
                real_sv.append(sv[i])
            b += sv_y[i]
            b -= np.sum(a * sv_y * K[ind[i]].reshape((-1,1))[alphas_sv.reshape((-1,1))]) # choice the a_i*y_i*K(x,x) corectspond with alpha_sv
            count +=1 # count how many time you accumulate b beacause b just only take 1 time to calculate
        b /= count # divice count to get b
        if self.C:
            self.real_sv = np.array(real_sv)
        print("%d support vectors out of %d points" % (len(real_sv),n_samples))
        self.a = a
        self.b = b 
        self.sv_y = sv_y
        self.sv = sv
        
        # Weight vector
        if self.kernel == 'linear':
            self.w = np.sum(self.a.reshape((-1,1)) * self.sv_y.reshape((-1,1)) * self.sv,axis = 0)
        else:
            self.w = None
        
    def project(self,X):
        '''
        Calculate new label with new features

        Arguments:
            X --- new features

        Return:
            y --- predicted calculation
        '''
        if self.kernel == 'linear':
            return X.dot(self.w) + self.b
        elif self.kernel == 'rbf' or self.kernel == 'gaussian':
            K = sklearn.metrics.pairwise.rbf_kernel(X,self.sv,self.gamma)
        elif self.kernel == 'polynomial':
            K = sklearn.metrics.pairwise.polynomial_kernel(X,self.sv,degree =  self.degree, gamma = self.gamma, coef0 = self.coef0)
        return np.sum((self.a * self.sv_y).T*K,axis =1) + self.b
        
    def predict(self,X):
        '''
        predict binary labels with signed
        Arguments:
            X --- new features

        Return:
            y --- predicted labels
        '''
        return np.sign(self.project(X))
