import numpy as np

def poslog(x):
    return 1 if np.log(x) < 1 else np.log(x)

def gitt_approx_lat(a, b, m, c = 1/4):
    n = a + b
    approx1 = m/(np.power(poslog(m), 1.5))
    innerLog = np.power(poslog(m/n), 0.5)
    approx2 = m/(n*innerLog)
    beta = c*np.min([approx1, approx2])
    beta = np.max([1, beta])
    return a/n + np.sqrt(2 * np.log(beta)/n)

def gittins_old(T, arms, armData):
    K = len(arms)
    assert K == len(armData)
    priors = [[0,0] for _ in range(K)]

    rewards = []
    start = 0
    #set priors with data, round robin sample arms which do not have data
    for i in range(K):
        data_i = armData[i]
        for d in data_i:
            priors[i][1-d] += 1
        if(priors[i][0] + priors[i][1] < 1):
            rew = np.random.binomial(1, arms[i])
            priors[i][1-rew] += 1
            rewards.append(rew)
            start += 1

    #run gittins
    for i in range(start, T):
        gitt_idxs = []
        for a,b in priors:
            n = a + b
            m = T - i
            innerLog = np.sqrt(np.max([0.0, np.log(m/n)]))
            outerLog = 0 if innerLog == 0 else np.max([0.0, np.log(m * innerLog/n)])
            idx = a/n + np.sqrt(2 * outerLog/n)
            gitt_idxs.append(idx)
        arm = np.argmax(gitt_idxs)
        rew = np.random.binomial(1, arms[arm])
        rewards.append(rew)
        priors[arm][1-rew] += 1

    return rewards

#gittins approach with prior data, artificially increases time horizon
#by amt of data points read to more accurately estimate log(t) term
def gittins_FS_fixT(T, arms, armData):
    K = len(arms)
    assert K == len(armData)
    priors = [[0,0] for _ in range(K)]

    rewards = []
    start = 0
    dataPts = 0
    #set priors with data, round robin sample arms which do not have data
    for i in range(K):
        data_i = armData[i]
        dataPts += len(data_i)
        for d in data_i:
            priors[i][1-d] += 1
        if(priors[i][0] + priors[i][1] < 1):
            rew = np.random.binomial(1, arms[i])
            priors[i][1-rew] += 1
            rewards.append(rew)
            start += 1

    start += dataPts
    T += dataPts
    #run gittins
    for i in range(start, T):
        gitt_idxs = []
        for a,b in priors:
            n = a + b
            approx1 = m/(np.power(np.log(m), 1.5))
            innerLog = np.power(np.log(m/n), 0.5)
            approx2 = m/(n*innerLog)
            beta = np.min([approx1, approx2])
            beta = np.max([1, beta])
            idx = a/n + np.sqrt(2 * np.log(beta)/n)
            gitt_idxs.append(idx)
        arm = np.argmax(gitt_idxs)
        rew = np.random.binomial(1, arms[arm])
        rewards.append(rew)
        priors[arm][1-rew] += 1

    return rewards

def gittins_FS(T, arms, armData):
    K = len(arms)
    assert K == len(armData)
    priors = [[0,0] for _ in range(K)]

    rewards = []
    start = 0
    dataPts = 0
    #set priors with data, round robin sample arms which do not have data
    for i in range(K):
        data_i = armData[i]
        dataPts += len(data_i)
        for d in data_i:
            priors[i][1-d] += 1
        if(priors[i][0] + priors[i][1] < 1):
            rew = np.random.binomial(1, arms[i])
            priors[i][1-rew] += 1
            rewards.append(rew)
            start += 1

    start += dataPts
    T += dataPts
    #run gittins
    for i in range(start, T):
        gitt_idxs = []
        for a,b in priors:
            gitt_idxs.append(gitt_approx_lat(a, b, T-i))
        arm = np.argmax(gitt_idxs)
        rew = np.random.binomial(1, arms[arm])
        rewards.append(rew)
        priors[arm][1-rew] += 1

    return rewards
