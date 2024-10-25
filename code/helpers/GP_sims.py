import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

class GPFunction:
    def __init__(self, base, sample_var, range, maxPoint):
        self.truef = base
        self.var = sample_var
        self.x_min = range[0]
        self.x_max = range[1]
        self.opt_x = maxPoint[0]
        self.opt = maxPoint[1]

    def sample(self, x):
        return self.truef(x) + np.random.normal(0, np.sqrt(self.var))

class GPRegression:
    def __init__(self, kernel, beta):
        self.kernel = kernel
        self.beta = beta #precision of noise

    def fit(self, X, t):
        '''
        Parameters
        ----------
        X : 2-D numpy array
            Array representing training input data, with X[n, i] being the i-th element of n-th point in X.
        t : 1-D numpy array
            Array representing training label data.
        '''
        self.X_train = X
        self.t_train = t

    def predict(self, X_test, return_cov=False):
        '''
        Parameters
        ----------
        X_test : 2-D numpy array
            Array representing test input data, with X_test[n, i] being the i-th element of n-th point in X.
        return_cov : bool
            If True, predictive standard covariance is also returned.

        Returns
        ----------
        mean : 1-D numpy array
            Array representing predictive mean.
        cov : 2-D numpy array, optional
            Array reprensenting predictive covariance, returned only if return_cov is True.
        '''
        Kinv = np.linalg.pinv(self.kernel(self.X_train, self.X_train) + (1.0/self.beta) * np.identity(len(self.X_train)))
        k = self.kernel(self.X_train, X_test)
        c = self.kernel(X_test,X_test)
        mean = k.T @ Kinv @ self.t_train
        if not(return_cov):
            return mean
        else:
            #cov = np.sqrt( self.kernel.diag(X_test) + 1.0/self.beta - np.diag( k.T @ Kinv @ k ) )
            cov = c - k.T @ Kinv @ k # + (1.0/self.beta)*np.identity(len(X_test))
            return mean, cov

def FS(T, func, prior_X, prior_y, gpr, sub_D = 100, show_plot =  False):
    X = np.array(prior_X)
    y = np.array(prior_y)
    reward_vec = []
    rng = np.random.default_rng()

    for i in tqdm(range(T), leave = False):
        v = 1
        D_t = np.random.uniform(func.x_min, func.x_max, sub_D)
        gpr.fit(np.reshape(X, (len(X), 1)), y)
        pred_mean, pred_cov = gpr.predict(np.reshape(D_t,(len(D_t),1)), return_cov=True)
        yy = rng.multivariate_normal(pred_mean, v*v*pred_cov, method="eigh")

        newIdx = np.argmax(yy)
        newX = D_t[newIdx]
        newY = func.sample(newX)
        X = np.append(X, newX)
        y = np.append(y, newY)

        reward_vec.append(newY)

    if(show_plot):
        N = len(prior_X)
        data_pts = [X[:N], y[:N]]
        samp_pts = [X[N:], y[N:]]
        plot_GP(func, gpr, data_pts, samp_pts)

    return reward_vec

def AR(T, func, prior_X, prior_y, gpr, sub_D = 100, show_plot = False):
    X = np.array([])
    y = np.array([])

    #sort data by X
    ind = np.argsort(prior_X)
    prior_X = np.array(prior_X)[ind]
    prior_y = np.array(prior_y)[ind]

    reward_vec = []
    data_vec = [[], []]
    rng = np.random.default_rng()

    for i in tqdm(range(T), leave = False):
        realPlay = False
        while(not realPlay):
            v = 1
            D_t = np.random.uniform(x_min, x_max, sub_D)
            D_t = np.sort(D_t)
            gpr.fit(np.reshape(X, (len(X), 1)), y)
            pred_mean, pred_cov = gpr.predict(np.reshape(D_t,(len(D_t),1)), return_cov=True)
            yy = rng.multivariate_normal(pred_mean, v*v*pred_cov, method="eigh")

            newIdx = np.argmax(yy)
            newX = D_t[newIdx]

            #AR Implementation: only use data point if it is closer to the sampling point newX than any other x in D_t
            realPlay = True
            if(len(prior_X) > 0):
                data_min = func.x_min if newIdx == 0 else (newX + D_t[newIdx-1])/2
                data_max = func.x_max if newIdx == sub_D-1 else (newX + D_t[newIdx+1])/2
                data_dist = np.array(prior_X) - newX
                below_idx = np.where(data_dist < 0, data_dist, -np.inf).argmax()
                below = prior_X[below_idx]
                if(below > data_min and below < data_max):
                    #if the data point below is within the data range, sample it
                    newX = below
                    newY = prior_y[below_idx]
                    prior_X = np.delete(prior_X, below_idx)
                    prior_y = np.delete(prior_y, below_idx)
                    realPlay = False
                else:
                    #try above point
                    if(below_idx < len(prior_X) - 1):
                        above = prior_X[below_idx + 1]
                        if(above > data_min and above < data_max):
                            newX = above
                            newY = prior_y[below_idx+1]
                            prior_X = np.delete(prior_X, below_idx+1)
                            prior_y = np.delete(prior_y, below_idx+1)
                            realPlay = False

            if(realPlay == True):
                newY = func.sample(newX)

            X = np.append(X, newX)
            y = np.append(y, newY)

        reward_vec.append(newY)
        if(i == 0):
            data_vec[0] = prior_X
        if(i == T-1):
            data_vec[1] = prior_X

    if(show_plot):
        N = len(prior_X)
        data_pts = [[], []]
        samp_pts = [X, y]
        plot_GP(func, gpr, data_pts, samp_pts)

    return reward_vec, data_vec

def AR_rad(T, func, prior_X, prior_y, gpr, data_rad, sub_D = 100, show_plot = False):
    X = np.array([])
    y = np.array([])

    #sort data by X
    ind = np.argsort(prior_X)
    prior_X = np.array(prior_X)[ind]
    prior_y = np.array(prior_y)[ind]

    reward_vec = []
    data_vec = [[], []]
    rng = np.random.default_rng()

    for i in tqdm(range(T), leave = False):
        realPlay = False
        while(not realPlay):
            v = 1
            D_t = np.random.uniform(func.x_min, func.x_max, sub_D)
            D_t = np.sort(D_t)
            gpr.fit(np.reshape(X, (len(X), 1)), y)
            pred_mean, pred_cov = gpr.predict(np.reshape(D_t,(len(D_t),1)), return_cov=True)

            yy = rng.multivariate_normal(pred_mean, v*v*pred_cov, method="eigh")

            newIdx = np.argmax(yy)
            newX = D_t[newIdx]

            #AR radius implementation: use a data point if there is one within a fixed radius of the point to be sampled
            realPlay = True
            if(len(prior_X) > 0):
                data_dist = np.abs(np.array(prior_X) - newX)
                closest_idx = data_dist.argmin()
                if(data_dist[closest_idx] <= data_rad):
                    #if the closest data point is within the data radius, sample it
                    newX = prior_X[closest_idx]
                    newY = prior_y[closest_idx]
                    prior_X = np.delete(prior_X, closest_idx)
                    prior_y = np.delete(prior_y, closest_idx)
                    realPlay = False

            if(realPlay == True):
                newY = func.sample(newX)

            X = np.append(X, newX)
            y = np.append(y, newY)
        reward_vec.append(newY)
        if(i == 0):
            data_vec[0] = prior_X
        if(i == T-1):
            data_vec[1] = prior_X

    if(show_plot):
        N = len(prior_X)
        data_pts = [[], []]
        samp_pts = [X, y]
        plot_GP(func, gpr, data_pts, samp_pts)

    return reward_vec, data_vec

def AR_lvl_set(T, func, prior_X, prior_y, gpr, num_sds, sub_D = 100, show_plot = False):
    X = np.array([])
    y = np.array([])

    reward_vec = []
    data_vec = [[], []]
    rng = np.random.default_rng()

    for i in tqdm(range(T), leave = False):
        realPlay = False
        while(not realPlay):
            v = 1
            D_t = np.random.uniform(func.x_min, func.x_max, sub_D)
            gpr.fit(np.reshape(X, (len(X), 1)), y)
            pred_mean, pred_cov = gpr.predict(np.reshape(D_t,(len(D_t),1)), return_cov=True)

            yy = rng.multivariate_normal(pred_mean, v*v*pred_cov, method="eigh")
            optIdx = np.argmax(yy)
            optY = np.max(yy)

            #AR level set: use data points if sampled reward is within dt of sampled opt reward
            #samples a random data point in the level set if there are multiple
            realPlay = True
            if(len(prior_X) > 0):
                data_dist = optY - yy
                peak_var = pred_cov[optIdx][optIdx]
                sample_range = data_dist < num_sds * np.sqrt(peak_var)

                for data_idx in range(len(prior_X)):
                    data_x = prior_X[data_idx]
                    closest_idx = np.argmin(np.abs(D_t - data_x))
                    if(sample_range[closest_idx]):
                        #if a data point is closest to a sample point in the level set, we can sample it
                        newX = data_x
                        newY = prior_y[data_idx]
                        prior_X = np.delete(prior_X, data_idx)
                        prior_y = np.delete(prior_y, data_idx)
                        realPlay = False
                        break


            if(realPlay == True):
                newX = D_t[optIdx]
                newY = func.sample(newX)

            X = np.append(X, newX)
            y = np.append(y, newY)
        reward_vec.append(newY)
        if(i == 0):
            data_vec[0] = prior_X
        if(i == T-1):
            data_vec[1] = prior_X

    if(show_plot):
        N = len(prior_X)
        data_pts = [[], []]
        samp_pts = [X, y]
        plot_GP(func, gpr, data_pts, samp_pts)

    return reward_vec, data_vec

def plot_GP(func, gpr, data_pts, samp_pts):
    Xcont = np.linspace(func.x_min, func.x_max, 200)
    Xtest = np.reshape(Xcont, (len(Xcont), 1))
    pred_mean, pred_cov = gpr.predict(Xtest, return_cov=True)
    plt.plot(data_pts[0], data_pts[1],'og',markersize=10,label='data read')
    plt.plot(samp_pts[0], samp_pts[1],'or',markersize=10,label='points sampled')
    plt.plot(Xcont, pred_mean, label='predictive mean')
    plt.plot(Xcont, func.truef(Xcont), ':r', label='ground truth')
    pred_stddev = np.sqrt(np.diagonal(pred_cov))
    upper_cb = pred_mean + 2*pred_stddev
    lower_cb = pred_mean - 2*pred_stddev
    plt.fill_between(Xcont, upper_cb, lower_cb, alpha=0.2)
    plt.figure()
