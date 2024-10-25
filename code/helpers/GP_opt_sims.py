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

def plot_GP(func, X, y, kern, K_inv):
    Xcont = np.linspace(func.x_min, func.x_max, 200)
    Xtest = np.reshape(Xcont, (len(Xcont), 1))
    k = kern(X, Xtest)
    c = kern(Xtest, Xtest)
    pred_mean = k.T @ K_inv @ y
    pred_cov = c - k.T @ K_inv @ k

    plt.plot(X, y,'og',markersize=10,label='data points')
    plt.plot(Xcont, pred_mean, label='predictive mean')
    plt.plot(Xcont, func.truef(Xcont), ':r', label='ground truth')
    pred_stddev = np.sqrt(np.diagonal(pred_cov))
    upper_cb = pred_mean.T[0] + 2*pred_stddev
    lower_cb = pred_mean.T[0] - 2*pred_stddev
    plt.fill_between(Xcont, upper_cb, lower_cb, alpha=0.2)
    plt.figure()

def FS(T, func, prior_X, prior_y, kern, beta, sub_D = 100, show_plot = False):
    reward_vec = []
    start = 0
    if(prior_X.shape[0] == 0):
        newX = np.random.uniform(func.x_min, func.x_max)
        newY = func.sample(newX)
        prior_X = [[newX]]
        prior_y = [[newY]]
        reward_vec.append(newY)
        start = 1

    X = np.array(prior_X)
    y = np.array(prior_y)

    rng = np.random.default_rng()

    #initialize kernel and inverse
    K = kern(X, X) + (1.0/beta)*np.identity(len(X))
    K_inv = np.linalg.pinv(K)

    for i in tqdm(range(start, T), leave = False):
        #compute predictive mean and cov for randomized sample points
        D_t = np.random.uniform(func.x_min, func.x_max, (sub_D, 1))
        k = kern(X, D_t)
        c = kern(D_t, D_t)
        pred_mean = k.T @ K_inv @ y
        pred_cov = c - k.T @ K_inv @ k

        #(thompson) sample function from GP
        v = 1
        yy = rng.multivariate_normal(pred_mean.T[0], v*v*pred_cov, method="eigh")

        #sample data point at maximum of sampled function
        newIdx = np.argmax(yy)
        newX = np.array([D_t[newIdx]])
        newY = func.sample(newX)[0][0]

        #update K and K_inv
        b = kern(X, newX)
        d = kern(newX, newX) + 1/beta
        e = K_inv @ b
        g = 1/(d - (b.T @ e))

        K = np.block([
            [K, b],
            [b.T, d]
        ])
        K_inv = np.block([
            [K_inv + g * (e @ e.T), -g * e],
            [-g * e.T, g]
        ])

        #update X, y and track reward
        X = np.block([[X],
                      [newX]])
        y = np.block([[y],
                      [np.array(newY)]])
        reward_vec.append(newY)

    if(show_plot):
        N = len(prior_X)
        data_pts = [X[:N], y[:N]]
        samp_pts = [X[N:], y[N:]]
        plot_GP(func, X, y, kern, K_inv)

    return reward_vec

def AR_rad(T, func, data_X, data_y, kern, beta, data_rad, sub_D = 100, show_plot = False):
    reward_vec = []
    data_vec = []
    rng = np.random.default_rng()

    #initialize
    start_X = np.random.uniform(func.x_min, func.x_max)
    i = 1
    if(len(data_X) > 0):
        data_dist = np.abs(np.array(data_X) - start_X)
        closest_idx = data_dist.argmin()
        if(data_dist[closest_idx] <= data_rad):
            #if the closest data point is within the data radius, sample it
            start_X = data_X[closest_idx]
            start_Y = data_y[closest_idx]
            data_X = np.delete(data_X, closest_idx)
            data_y = np.delete(data_y, closest_idx)
            i = 0
        else:
            start_Y = func.sample(start_X)
            reward_vec.append(start_Y)
    else:
        start_Y = func.sample(start_X)
        reward_vec.append(start_Y)

    X = np.full((1,1), start_X)
    y = np.full((1,1), start_Y)

    #initial K will always be 1x1
    K = kern(X, X) + (1.0/beta)
    K_inv = 1/K

    while(i < T):
        D_t = np.random.uniform(func.x_min, func.x_max, (sub_D, 1))
        k = kern(X, D_t)
        c = kern(D_t, D_t)
        pred_mean = k.T @ K_inv @ y
        pred_cov = c - k.T @ K_inv @ k

        #(thompson) sample function from GP
        v = 1
        yy = rng.multivariate_normal(pred_mean.T[0], v*v*pred_cov, method="eigh")

        #sample data point at maximum of sampled function
        newIdx = np.argmax(yy)
        newX = D_t[newIdx][0]

        if(len(data_X) > 0):
            data_dist = np.abs(np.array(data_X) - newX)
            closest_idx = data_dist.argmin()
            if(data_dist[closest_idx] <= data_rad):
                #if the closest data point is within the data radius, sample it
                newX = data_X[closest_idx][0]
                newY = data_y[closest_idx][0]
                data_X = np.delete(data_X, closest_idx)
                data_y = np.delete(data_y, closest_idx)
            else:
                newY = func.sample(newX)
                reward_vec.append(newY)
                i += 1
        else:
            newY = func.sample(newX)
            reward_vec.append(newY)
            i += 1

        #update K and K_inv
        newX = np.full((1,1), newX)
        newY = np.full((1,1), newY)
        b = kern(X, newX)
        d = kern(newX, newX) + 1/beta
        e = K_inv @ b
        g = 1/(d - (b.T @ e))

        K = np.block([
            [K, b],
            [b.T, d]
        ])
        K_inv = np.block([
            [K_inv + g * (e @ e.T), -g * e],
            [-g * e.T, g]
        ])

        #update X, y
        X = np.block([[X],
                      [newX]])
        y = np.block([[y],
                      [newY]])

    if(show_plot):
        plot_GP(func, X, y, kern, K_inv)

    return reward_vec, data_X

def AR_lvl_set(T, func, data_X, data_y, kern, beta, num_sds, sub_D = 100, show_plot = False):
    reward_vec = []
    data_vec = []
    rng = np.random.default_rng()

    #initialize - pick a data point at random
    if(len(data_X) > 0):
        i = 0
        rand_idx = np.random.randint(len(data_X))
        start_X = data_X[rand_idx]
        start_Y = data_y[rand_idx]
        data_X = np.delete(data_X, rand_idx)
        data_y = np.delete(data_y, rand_idx)

    else:
        i = 1
        start_X = np.random.uniform(func.x_min, func.x_max)
        start_Y = func.sample(start_X)
        reward_vec.append(start_Y)

    X = np.full((1,1), start_X)
    y = np.full((1,1), start_Y)
    #initial K will always be 1x1
    K = kern(X, X) + (1.0/beta)
    K_inv = 1/K

    while(i < T):
        D_t = np.random.uniform(func.x_min, func.x_max, (sub_D, 1))
        k = kern(X, D_t)
        c = kern(D_t, D_t)
        pred_mean = k.T @ K_inv @ y
        pred_cov = c - k.T @ K_inv @ k

        #(thompson) sample function from GP
        v = 1
        yy = rng.multivariate_normal(pred_mean.T[0], v*v*pred_cov, method="eigh")

        #sample data point at maximum of sampled function
        optIdx = np.argmax(yy)

        #AR level set: use data points if sampled reward is within dt of sampled opt reward
        #samples a random data point in the level set if there are multiple
        if(len(data_X) > 0):
            optY = np.max(yy)
            data_dist = optY - yy
            peak_var = pred_cov[optIdx][optIdx]
            sample_range = data_dist < num_sds * np.sqrt(peak_var)

            in_lvl_set = []
            for data_idx in range(len(data_X)):
                pt_x = data_X[data_idx]
                closest_idx = np.argmin(np.abs(D_t - pt_x))
                if(sample_range[closest_idx]):
                    #if a data point is closest to a sample point in the level set, we can sample it
                    in_lvl_set.append(data_idx)

            if(len(in_lvl_set) > 0):
                samp_idx = np.random.choice(in_lvl_set)
                newX = data_X[samp_idx]
                newY = data_y[samp_idx]
                data_X = np.delete(data_X, samp_idx)
                data_y = np.delete(data_y, samp_idx)

            else:
                newX = D_t[optIdx][0]
                newY = func.sample(newX)
                reward_vec.append(newY)
                i += 1

        else:
            newX = D_t[optIdx][0]
            newY = func.sample(newX)
            reward_vec.append(newY)
            i += 1

        newX = np.full((1,1), newX)
        newY = np.full((1,1), newY)

        #update K and K_inv
        b = kern(X, newX)
        d = kern(newX, newX) + 1/beta
        e = K_inv @ b
        g = 1/(d - (b.T @ e))

        K = np.block([
            [K, b],
            [b.T, d]
        ])
        K_inv = np.block([
            [K_inv + g * (e @ e.T), -g * e],
            [-g * e.T, g]
        ])

        #update X, y
        X = np.block([[X],
                      [newX]])
        y = np.block([[y],
                      [np.array(newY)]])

    if(show_plot):
        data_pts = [[], []]
        samp_pts = [X, y]
        plot_GP(func, X, y, kern, K_inv)

    return reward_vec, data_X
