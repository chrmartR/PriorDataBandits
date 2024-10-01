import numpy as np

class MyKernel:
    def __init__(self, theta, bounds=None):
        self.theta = theta
        self.bounds = bounds

    def __call__(self, X, Y, eval_gradient=False):
        '''
        Compute kernel matrix for two data vectors

        Parameters
        ----------
        X, Y : 2-D numpy array
            Input points. X[n, i] (resp. Y[n, i]) = i-th element of n-th point in X (resp Y).
        eval_gradient : bool
            If True, return gradient of kernel matrix w.r.t. to hyperparams

        Returns
        ----------
        K : 2-D numpy array, shape = (len(X), len(Y))
            Kernel matrix K[i, j]= k(X[i], Y[j])
        gradK : 3-D numpy array, shape = (len(self.theta), len(X), len(Y)), optional
            Gradient of kernel matrix. gradK[i, m, n] = derivative of K[m, n] w.r.t. self.theta[i]
            Returned only if return_std is True.
        '''

        tmp = np.reshape(np.sum(X**2,axis=1), (len(X), 1)) + np.sum(Y**2, axis=1)  -2 * (X @ Y.T)
        K = self.theta[0]*np.exp(-self.theta[1]/2*tmp) + self.theta[2] + self.theta[3]*(X @ Y.T)

        if not(eval_gradient):
            return K
        else:
            gradK = np.zeros((len(self.theta), len(X), len(Y)))
            gradK[0] = np.exp(-self.theta[1]/2*tmp)
            gradK[1] = -self.theta[0]/2*tmp*np.exp(-self.theta[1]/2*tmp)
            gradK[2] = np.ones((len(X), len(Y)))
            gradK[3] = X @ Y.T
            return K, gradK

    def diag(self, X):
        '''
        Return diagonal elements of kernel matrix.

        Parameters
        ----------
        X : 2-D numpy array
            numpy array representing input points. X[n, i] represents the i-th element of n-th point in X.

        Returns
        ----------
        diagK : 1-D numpy array
            numpy array representing the diagonal elements of the kernel matrix. diagK[n] = K[n, n]
        '''
        diagK = self.theta[0] + self.theta[2] + self.theta[3]*np.sum(X**2, axis=1)
        return diagK

class OUKernel:
    def __init__(self, theta, bounds=None):
        self.theta = theta
        self.bounds = bounds

    def __call__(self, X, Y):
        '''
        Compute the OU kernel matrix for two data vectors

        Parameters
        ----------
        X, Y : 2-D numpy array
            Input points. X[n, i] (resp. Y[n, i]) = i-th element of n-th point in X (resp Y).
        eval_gradient : bool
            If True, return gradient of kernel matrix w.r.t. to hyperparams

        Returns
        ----------
        K : 2-D numpy array, shape = (len(X), len(Y))
            Kernel matrix K[i, j]= k(X[i], Y[j])
        gradK : 3-D numpy array, shape = (len(self.theta), len(X), len(Y)), optional
            Gradient of kernel matrix. gradK[i, m, n] = derivative of K[m, n] w.r.t. self.theta[i]
            Returned only if return_std is True.
        '''

        norms_X = (X ** 2).sum(axis=1)
        norms_Y = (Y ** 2).sum(axis=1)
        tmp = norms_X.reshape(-1, 1) + norms_Y - 2 * np.dot(X, Y.T)
        K = self.theta[0]*np.exp(-self.theta[1]*np.sqrt(tmp))

        return K

class RBFKernel:
    def __init__(self, gamma, bounds=None):
        self.gamma = gamma
        self.bounds = bounds

    def __call__(self, X, Y):
        '''
        Compute the RBF kernel matrix for two data vectors

        Parameters
        ----------
        X, Y : 2-D numpy array
            Input points. X[n, i] (resp. Y[n, i]) = i-th element of n-th point in X (resp Y).
        eval_gradient : bool
            If True, return gradient of kernel matrix w.r.t. to hyperparams

        Returns
        ----------
        K : 2-D numpy array, shape = (len(X), len(Y))
            Kernel matrix K[i, j]= k(X[i], Y[j])
        gradK : 3-D numpy array, shape = (len(self.theta), len(X), len(Y)), optional
            Gradient of kernel matrix. gradK[i, m, n] = derivative of K[m, n] w.r.t. self.theta[i]
            Returned only if return_std is True.
        '''

        norms_X = (X ** 2).sum(axis=1)
        norms_Y = (Y ** 2).sum(axis=1)
        tmp = norms_X.reshape(-1, 1) + norms_Y - 2 * np.dot(X, Y.T)
        K = np.exp(-self.gamma*tmp)

        return K
