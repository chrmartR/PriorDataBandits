import numpy as np

#priors are [mean, precision], where precision = 1/var
#we assume a known precision of TAU = 1 for each arm (can be specified)

def FS(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    rewards = []
    start = 0

    #set priors with data
    for i in range(K):
        data_i = armData[i]
        numerators[i] += TAU * np.sum(data_i)
        denominators[i] += TAU * len(data_i)

        #sample arms which do not have data (initial round robin)
        if(denominators[i] <= 0):
            rew = np.random.normal(arms[i], np.sqrt(1/TAU))
            numerators[i] += TAU * rew
            denominators[i] += TAU
            rewards.append(rew)
            start += 1

        priors[i][0] = numerators[i]/denominators[i]
        priors[i][1] = denominators[i]


    for i in range(start, T):
        samples = [np.random.normal(m,np.sqrt(1/t)) for m,t in priors]
        arm = np.argmax(samples)
        rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]
        rewards.append(rew)

    return rewards

def AR(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)
    armData = [list(data_i) for data_i in armData] #prevent changing data

    rewards = []
    countReal = 0

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    #set priors with data
    for arm in range(K):
        #sample arms which do not have data (initial round robin)
        if(denominators[arm] <= 0):
            if(len(armData[arm]) > 0):
                rew = armData[arm].pop(0)
            else:
                rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
                rewards.append(rew)
                countReal += 1
            numerators[arm] += TAU * rew
            denominators[arm] += TAU
            priors[arm][0] = numerators[arm]/denominators[arm]
            priors[arm][1] = denominators[arm]

    while countReal < T:
        samples = [np.random.normal(m,np.sqrt(1/t)) for m,t in priors]
        arm = np.argmax(samples)
        if(len(armData[arm]) > 0):
            rew = armData[arm].pop(0)
        else:
            rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
            rewards.append(rew)
            countReal += 1
        #update prior
        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]
    return rewards

def flat_priors(K):
    return [[0.5, 0] for _ in range(K)]

#to start with flat prior, have TAU = 0 (so prior=[0.5, 0])
#priors listed as [mean, precision]
def UCB_FS(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)

    rewards = []
    start = 0
    dataPts = 0

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    #set priors with data
    for arm in range(K):
        data_i = armData[arm]
        dataPts += len(data_i)
        numerators[arm] += TAU * np.sum(data_i)
        denominators[arm] += TAU * len(data_i)

        #sample arms which do not have data (initial round robin)
        if(denominators[arm] <= 0):
            rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
            numerators[arm] += TAU * rew
            denominators[arm] += TAU
            rewards.append(rew)
            start += 1
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]

    #run UCB
    start += dataPts
    T += dataPts
    for i in range(start, T):
        UCB = [u + np.sqrt(2 * np.log(i+1)/t) for u,t in priors]
        arm = np.argmax(UCB)
        rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
        rewards.append(rew)
        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]

    return rewards

#to start with flat prior, have TAU = 0 (so prior=[0.5, 0])
#priors listed as [mean, precision]
def UCB_FS_monotone(T, arms, priors, armData, TAU = 1):
    K = len(arms)
    assert K == len(armData)
    assert K == len(priors)

    rewards = []
    start = 0
    dataPts = 0

    numerators = [m * t for m,t in priors]
    denominators = [t for _,t in priors]

    #set priors with data
    for arm in range(K):
        data_i = armData[arm]
        dataPts += len(data_i)
        numerators[arm] += TAU * np.sum(data_i)
        denominators[arm] += TAU * len(data_i)

        #sample arms which do not have data (initial round robin)
        if(denominators[arm] <= 0):
            rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
            numerators[arm] += TAU * rew
            denominators[arm] += TAU
            rewards.append(rew)
            start += 1
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]

    #run UCB
    UCB_min = [u + np.sqrt(2) for u,t in priors]
    start += dataPts
    T += dataPts
    for i in range(start, T):
        UCB = [u + np.sqrt(2 * np.log(i+1)/t) for u,t in priors]
        UCB_min = [np.min([UCB_min[i], UCB[i]]) for i in range(K)]
        arm = np.argmax(UCB_min)
        rew = np.random.normal(arms[arm], np.sqrt(1/TAU))
        rewards.append(rew)
        numerators[arm] += TAU * rew
        denominators[arm] += TAU
        priors[arm][0] = numerators[arm]/denominators[arm]
        priors[arm][1] = denominators[arm]

    return rewards
