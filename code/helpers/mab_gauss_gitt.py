import numpy as np

def flat_priors(K):
    return [[0.5, 0] for _ in range(K)]

def poslog(x):
    return 1 if np.log(x) < 1 else np.log(x)

def gitt_approx_norm(u, t, m, c = 1/4, TAU = 1):
    #for a initially flat prior, the number of times an arm has been sampled
    #is equal to the precision of its posterior
    n = t/TAU
    approx1 = m/(np.power(poslog(m), 1.5))
    innerLog = np.power(poslog(m/n), 0.5)
    approx2 = m/(n*innerLog)
    beta = c*np.min([approx1, approx2])
    beta = np.max([1, beta])
    return u + np.sqrt(2 * np.log(beta)/n)

#priors listed as [mean, precision]
def gittins_FS(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    rewards = []
    start = 0
    dataPts = 0
    #set priors with data
    for arm in range(K):
        data_i = armData[arm]
        dataPts += len(data_i)
        numerators[arm] += TAU * np.sum(data_i)
        denominators[arm] += TAU * len(data_i)

        #sample arms which do not have data (initial round robin)
        if(denominators[arm] < 1):
            rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
            numerators[arm] += TAU * rew
            denominators[arm] += TAU
            rewards.append(rew)
            start += 1

        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]

    start += dataPts
    T += dataPts
    #run gittins
    for i in range(start, T):
        gitt_idxs = [gitt_approx_norm(m, t, T-i, TAU=TAU) for m,t in priors]
        arm = np.argmax(gitt_idxs)
        rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
        rewards.append(rew)
        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]

    return rewards

#priors listed as [mean, precision]
def gittins_AR(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)
    armData = [list(data_i) for data_i in armData] #prevent changing data

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    rewards = []
    i = 0
    #set priors with data
    for arm in range(K):
        #sample arms which do not have data (initial round robin)
        if(denominators[arm] <= 0):
            if(len(armData[arm]) > 0): #sample data
                rew = armData[arm].pop(0)
                T += 1 #for fake data, extend time horizon
            else: #sample true arm
                rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
                rewards.append(rew)

            numerators[arm] += TAU * rew
            denominators[arm] += TAU
            priors[arm][0] = numerators[arm]/denominators[arm]
            priors[arm][1] = denominators[arm]

            i += 1
    #run gittins
    while i < T:
        gitt_idxs = [gitt_approx_norm(m, t, T-i, TAU=TAU) for m,t in priors]
        arm = np.argmax(gitt_idxs)
        if(len(armData[arm]) > 0): #sample data
            rew = armData[arm].pop(0)
            T += 1 #for fake data, extend time horizon
        else: #sample true arm
            rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
            rewards.append(rew)
        i += 1

        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]


    return rewards

#priors listed as [mean, precision]
def gittins_FS_monotone(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    rewards = []
    horizon = T
    min_gitt_idxs = [1000 for _,_ in priors]
    #set priors with data
    for arm in range(K):
        data_i = armData[arm]
        for data in data_i:
            numerators[arm] += TAU * data
            denominators[arm] += TAU
            gitt_arm = gitt_approx_norm(numerators[arm]/denominators[arm], denominators[arm], horizon, TAU=TAU)
            min_gitt_idxs[arm] = np.min([min_gitt_idxs[arm], gitt_arm])

        #sample arms which do not have data (initial round robin)
        if(denominators[arm] < 1):
            rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
            numerators[arm] += TAU * rew
            denominators[arm] += TAU
            rewards.append(rew)
            horizon -= 1

        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]

    gitt_idxs = [gitt_approx_norm(m, t, horizon, TAU=TAU) for m,t in priors]
    min_gitt_idxs = [np.min([min_gitt_idxs[a], gitt_idxs[a]]) for a in range(K)]
    #run gittins
    while horizon > 0
        gitt_idxs = [gitt_approx_norm(m, t, horizon, TAU=TAU) for m,t in priors]
        min_gitt_idxs = [np.min([min_gitt_idxs[a], gitt_idxs[a]]) for a in range(K)]
        arm = np.argmax(min_gitt_idxs)
        rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
        rewards.append(rew)
        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]
        horizon -= 1

    return rewards

#priors listed as [mean, precision]
def gittins_AR_monotone(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)
    armData = [list(data_i) for data_i in armData] #prevent changing data

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    rewards = []
    i = 0
    #set priors with data
    for arm in range(K):
        #sample arms which do not have data (initial round robin)
        if(denominators[arm] <= 0):
            if(len(armData[arm]) > 0): #sample data
                rew = armData[arm].pop(0)
                T += 1 #for fake data, extend time horizon
            else: #sample true arm
                rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
                rewards.append(rew)


            numerators[arm] += TAU * rew
            denominators[arm] += TAU
            priors[arm][0] = numerators[arm]/denominators[arm]
            priors[arm][1] = denominators[arm]

            i += 1

    min_gitt_idxs = [gitt_approx_norm(m, t, T-i, TAU=TAU) for m,t in priors]
    #run gittins
    while i < T:
        gitt_idxs = [gitt_approx_norm(m, t, T-i, TAU=TAU) for m,t in priors]
        min_gitt_idxs = [np.min([min_gitt_idxs[a], gitt_idxs[a]]) for a in range(K)]
        arm = np.argmax(min_gitt_idxs)
        if(len(armData[arm]) > 0): #sample data
            rew = armData[arm].pop(0)
            T += 1 #for fake data, extend time horizon
        else: #sample true arm
            rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
            rewards.append(rew)
        i += 1

        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]


    return rewards
