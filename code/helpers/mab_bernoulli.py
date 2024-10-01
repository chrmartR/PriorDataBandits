import numpy as np

def gen_data(mean, amt):
    return np.random.binomial(1, mean, amt)

def FS_path(T, arms, priors, armData):
  K = len(arms)
  assert K == len(priors)
  assert K == len(armData)

  priors = np.array(priors)

  #read all data at start and update priors
  for arm in range(K):
      d = armData[arm]
      t_rew = np.sum(d)
      priors[arm][0] += t_rew
      priors[arm][1] += len(d) - t_rew

  rewards = []
  for i in range(T):
    samples = [np.random.beta(a,b) for a,b in priors]
    arm = np.argmax(samples)
    rew = np.random.binomial(1, arms[arm])
    priors[arm][0] += rew
    priors[arm][1] += 1 - rew

    rewards.append(rew)
  return rewards

def FS(T, arms, priors, armData):
    K = len(arms)
    assert K == len(priors)
    assert K == len(armData)

    priors = np.array(priors)

    #read all data at start and update priors
    for arm in range(K):
        d = armData[arm]
        t_rew = np.sum(d)
        priors[arm][0] += t_rew
        priors[arm][1] += len(d) - t_rew

    total = 0
    for i in range(T):
        samples = [np.random.beta(a,b) for a,b in priors]
        arm = np.argmax(samples)
        rew = np.random.binomial(1, arms[arm])
        priors[arm][0] += rew
        priors[arm][1] += 1 - rew

        total += rew
    return total

def AR_path(T, arms, priors, armData):
  K = len(arms)
  assert K == len(armData)

  readIdx = [0 for _ in range(K)]

  rewards = []
  countReal = 0

  while countReal < T:
      samples = [np.random.beta(a,b) for a,b in priors]
      arm = np.argmax(samples)
      if(readIdx[arm] < len(armData[arm])):
          rew = armData[arm][readIdx[arm]]
          readIdx[arm] += 1
      else:
          rew = np.random.binomial(1, arms[arm])
          rewards.append(rew)
          countReal += 1
      #update prior
      priors[arm][0] += rew
      priors[arm][1] += 1 - rew
  return rewards

def AR(T, arms, priors, armData):
    K = len(arms)
    assert K == len(armData)

    priors = np.array(priors)
    readIdx = [0 for _ in range(K)]

    total = 0
    countReal = 0

    while countReal < T:
        samples = [np.random.beta(a,b) for a,b in priors]
        arm = np.argmax(samples)
        if(readIdx[arm] < len(armData[arm])):
            rew = armData[arm][readIdx[arm]]
            readIdx[arm] += 1
        else:
            rew = np.random.binomial(1, arms[arm])
            total += rew
            countReal += 1
        #update prior
        priors[arm][0] += rew
        priors[arm][1] += 1 - rew
    return total

def flat_priors(K):
    return [[1, 1] for _ in range(K)]

def UCB_FS(T, arms, armData):
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

  #run UCB
  start += dataPts
  T += dataPts
  for i in range(start, T):
    UCB = [(a/(a+b)) + np.sqrt(2 * np.log(i+1)/(a+b)) for a,b in priors]
    arm = np.argmax(UCB)
    rew = np.random.binomial(1, arms[arm])
    rewards.append(rew)
    priors[arm][1-rew] += 1

  return rewards

def poslog(x):
    return 1 if np.log(x) < 1 else np.log(x)

def gitt_approx_norm(a, b, m, c = 1/4):
    n = a + b
    approx1 = m/(np.power(poslog(m), 1.5))
    innerLog = np.power(poslog(m/n), 0.5)
    approx2 = m/(n*innerLog)
    beta = c*np.min([approx1, approx2])
    beta = np.max([1, beta])
    return a/n + np.sqrt(2 * np.log(beta)/n)

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
            gitt_idxs.append(gitt_approx_norm(a, b, T-i))
        arm = np.argmax(gitt_idxs)
        rew = np.random.binomial(1, arms[arm])
        rewards.append(rew)
        priors[arm][1-rew] += 1

    return rewards
