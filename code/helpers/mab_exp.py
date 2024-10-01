import numpy as np

def flat_priors(K):
    return [[0, 0] for _ in range(K)]

def gen_data(scale, amt):
    return np.random.exponential(scale = scale, size = amt)

#priors are [alpha = shape, beta = rate] (mean a/b)
#usually will have improper flat priors [0, 0]
def FS(T, arms, priors, armData):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)

    rewards = []
    start = 0

    #set priors with data
    for arm in range(K):
        data_i = armData[arm]
        priors[arm][0] += len(data_i)
        priors[arm][1] += np.sum(data_i)

        #sample arms which do not have data (initial round robin)
        if(priors[arm][0] <= 0):
            rew = np.random.exponential(arms[arm])
            priors[arm][0] += 1
            priors[arm][1] += rew
            rewards.append(rew)
            start += 1

    for i in range(start, T):
        samples = [np.random.gamma(a,1/b) for a,b in priors]
        arm = np.argmin(samples)
        rew = np.random.exponential(arms[arm])
        priors[arm][0] += 1
        priors[arm][1] += rew
        rewards.append(rew)
    #print(priors)
    return rewards

def AR(T, arms, priors, armData):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)
    armData = [list(data_i) for data_i in armData] #prevent changing data

    rewards = []
    countReal = 0

    #sample arms which do not have data (initial round robin)
    #in case of improper flat priors
    for arm in range(K):
        if(priors[arm][0] <= 0):
            if(len(armData[arm]) > 0):
                rew = armData[arm].pop(0)
            else:
                rew = np.random.exponential(arms[arm])
                rewards.append(rew)
                countReal += 1
            priors[arm][0] += 1
            priors[arm][1] += rew

    while countReal < T:
        samples = [np.random.gamma(a,1/b) for a,b in priors]
        arm = np.argmin(samples)
        if(len(armData[arm]) > 0):
            rew = armData[arm].pop(0)
        else:
            rew = np.random.exponential(arms[arm])
            rewards.append(rew)
            countReal += 1
        #update prior
        priors[arm][0] += 1
        priors[arm][1] += rew
    #print(priors)
    return rewards
