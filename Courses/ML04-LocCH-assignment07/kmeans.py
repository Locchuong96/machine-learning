# set random seed so output is all name
np.random.seed(1)

class KMeans(object):
    def __init__(self): # No need to implement
        pass
    def pairwise_dist(self,x1,x2):
        '''
        Arguments:
            x1 --- (NxD) numpy array have D dimensions
            x2 --- (MxD) numpy array have D dimensions
        Return:
            dist --- (NxM) array, where dist2[i:j] is the euclidean distance between x[i,:] and y[j,:]
        '''
        x1_sum_square = np.sum(np.square(x1),axis = 1)
        x2_sum_square = np.sum(np.square(x2),axis = 1)
        mul = np.dot(x1,x2.T)
        dists = np.sqrt(abs(x1_sum_square[:,np.newaxis] + x2_sum_square - 2 * mul))
        return dists
    def _init_centers(self,points,K,**kwargs):
        '''
        Arguments:
            points --- (NxD) numpy array, where N is number of points and D is dimensionality
            K --- number of clusters
            kwargs --- any additional arguments you want
        Return:
            centers: K x D numpy array, the centers.
        '''
        N,D= points.shape
        centers = np.empty([K,D])
        for number in range(K):
            rand_index = np.random.randint(N)
            centers[number] = points[rand_index] # random choice in points with rand_index
        return centers
    def _update_assignment(self,centers,points):
        '''
        Arguments:
            centers: (KxD) numpy array, where K is the number of clusters, and D is dimensionaly
            points: (NxD) numpy array, the observations

        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point
        '''
        N,D = points.shape
        distances = self.pairwise_dist(points,centers)
        cluster_idx = np.argmin(distances,axis =1)
        return cluster_idx
    def _update_centers(self,old_centers,cluster_idx,points):
        '''
        Arguments:
            old_centers --- old centers KxD numpy array,where K is the number of clusters, and D is the dimension
            cluster_idx --- numpy array of length N, cluster assignment for each point
            points --- (NxD) numpy array,the observations
        Return:
            new_centers --- new centers, (KxD) numpy array, where K is the number of cluster
        '''
        K,D = old_centers.shape
        new_centers = np.empty(old_centers.shape)
        for i in range(K):
            new_centers[i] = np.mean(points[cluster_idx == i],axis = 0) # find mean of points which have cluster_idx == label
        return new_centers
    def _get_loss(self,centers,cluster_idx,points):
        '''
        Arguments:
            centers --- (KxD) numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx --- numpy array of length N, the cluster assignment for each point
            points --- (NxD) numpy array, the observations
        Return:
            loss --- a single float number, which is the objective function of KMeans
        '''
        dists = self.pairwise_dist(points,centers)
        loss = 0.0
        N,D = points.shape
        for i in range(N):
            loss += np.square(dists[i][cluster_idx[i]])
        return loss
    def __call__(self,points,K,max_iters = 100,abs_tol = 1e-16,rel_tol = 1e-16,verbose = False,**kwargs):
        '''
        Arguments:
            points --- (NxD) numpy array, where N is number points and D is the dimensionality
            K --- number of clusters
            max_iters --- maximum number of iterations
            abs_tol --- convergence criteria w.r.t absolute change of loss
            rel_tol --- convergence criteria w.r.t relative change of loss
            verbose --- boolean option to set whether method should print loss
            kwargs --- any additional arguments you want
        Return:
            cluster_assignments --- (Nx1) int number array
            cluster_centers --- (KxD) numpy array, the centers
            loss --- loss values of the objective function of KMeans
        '''
        centers = self._init_centers(points,K,**kwargs)
        loss = []
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers,points)
            centers = self._update_centers(centers,cluster_idx,points)
            loss.append(self._get_loss(centers,cluster_idx,points))
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss[-1])
                if diff < abs_tol and diff/prev_loss < rel_tol:
                    break
            prev_loss = loss[-1]
            if verbose:
                print('iter %d, loss: %.4f' %(it,loss[-1]))
        # remember the cluster_idx,centers,and loss
#         self.cluster_idx = cluster_idx
#         self.centers = centers
#         self.loss = loss
        return cluster_idx,centers,loss
    
    def predict(self,image_flatten,cluster_idx,centers):
        '''
        Arguments
            image_flatten --- (numpy array) image flatten
            cluster_idx --- cluster labels
            centers --- (K,D) centers array 
        
        Return:
            color label clustered
        '''
        updated_image_values =np.copy(image_flatten)
        distances = cdist(image_flatten,centers,'euclidean')
        y_pred = np.argmin(distances,axis = 1)
        for i in range(len(y_pred)):
            updated_image_values[i] = centers[y_pred[i]]
        updated_image_values = updated_image_values.reshape(r,c,ch)
        return updated_image_values
    
    def find_optimal_num_clusters(self,data,max_K = 15):
        '''
        Plots loss values for different number of clusters in K-means
        Arguments:
            image: input image of shape(H,W,3)
            max_K: number of clusters
        Return:
            None (plot loss values against number of clusters)
        '''
        y_val = []#np.empty(max_K)
        for i in range(max_K):
            cluster_idx,centers,loss = KMeans()(data,i+1) #cluster_idx, centers, y_val[i] = KMeans()(data, i + 1)
            y_val.append(loss[-1]) # y_val[i] = loss[-1]
            print(f"iter: {i}, loss: {y_val[-1]}")
        plt.plot(np.arange(max_K) + 1,y_val)
        plt.show()
        return y_val