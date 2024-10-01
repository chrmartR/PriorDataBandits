import numpy as np
from scipy.stats import norm

#approximate the probability of choosing each arm with TS (normal approx)
def approxProbs(fs_prior):
    assert len(fs_prior) == 2
    arm0_mean = fs_prior[0][0]/(fs_prior[0][0] + fs_prior[0][1])
    arm0_var = (fs_prior[0][0]*fs_prior[0][1])/((fs_prior[0][0]+fs_prior[0][1])*(fs_prior[0][0]+fs_prior[0][1])*(1 + fs_prior[0][0]+fs_prior[0][1]))
    arm1_mean = fs_prior[1][0]/(fs_prior[1][0] + fs_prior[1][1])
    arm1_var = (fs_prior[1][0]*fs_prior[1][1])/((fs_prior[1][0]+fs_prior[1][1])*(fs_prior[1][0]+fs_prior[1][1])*(1 + fs_prior[1][0]+fs_prior[1][1]))
    #because normal - normal = normal, we can estimate the distribution of sample 1 - sample 0
    prob_0 = norm(arm1_mean - arm0_mean, np.sqrt(arm0_var + arm1_var)).cdf(0)
    return [prob_0, 1-prob_0]

#approximate the probability of choosing each arm with AR (normal approx) by assuming arm sampling probabilities do not change
def AR_approxProbs(fs_prior, data):
    probs = approxProbs(fs_prior)
    amt0 = len(data[0])
    amt1 = len(data[1])
    if(amt0 == 0):
        if(amt1 == 0):
            return probs
        else:
            #arm 0 has no data, arm 1 has some data
            arm1prob = np.power(probs[1], amt1 + 1)
            return [1-arm1prob, arm1prob]
    else:
        if(amt1 == 0):
            arm0prob = np.power(probs[0], amt0 + 1)
            return [arm0prob, 1-arm0prob]
        else:
            #should never happen
            print("error")
            return probs

#approximate the probabilities of choosing each arm in AR with sampling
def AR_sampleProbs(ar_prior, remaining_data):
    K = len(ar_prior)
    assert K == 2
    SAMPLE_SIZE = 20
    chosen_arms = np.zeros(len(ar_prior))
    data = remaining_data[:]
    prior = np.copy(ar_prior)
    
    for i in range(SAMPLE_SIZE):
        realMove = False
        ar_arm = -1
        while not realMove:
            ar_samples = [np.random.beta(prior[j][0],prior[j][1]) for j in range(K)]
            ar_arm = np.argmax(ar_samples)
            if(len(data[ar_arm]) > 0):
                data_rew = data[ar_arm].pop()
                prior[ar_arm][0] += data_rew
                prior[ar_arm][1] += 1 - data_rew
            else:
                realMove = True
        chosen_arms[ar_arm] += 1
    #cannot divide-by-zero, so approximate
    MIN_PROB = 0.01
    probs = chosen_arms / SAMPLE_SIZE
    for i in range(K):
        if(probs[i] <= 0):
            probs[i] += MIN_PROB * (1 + 1/(K - 1))
            probs -= MIN_PROB / (K - 1)
    return probs

#hedge between AR and FS
def HEDGE(T, arms, armData):
    K = len(arms)
    assert (K == 2)
    ar_prior = np.ones((K , 2))
    fs_prior = [[1 + np.sum(d), 1 + len(d) - np.sum(d)] for d in armData]

    ar_weight = 1
    fs_weight = 1
    rewards = []

    for i in range(T):
        #sample FS and AR arm choices randomly
        fs_samples = [np.random.beta(fs_prior[j][0],fs_prior[j][1]) for j in range(K)]
        fs_arm = np.argmax(fs_samples)

        realMove = False
        ar_arm = -1
        while not realMove:
            ar_samples = [np.random.beta(ar_prior[j][0],ar_prior[j][1]) for j in range(K)]
            ar_arm = np.argmax(ar_samples)
            if(len(armData[ar_arm]) > 0):
                data_rew = armData[ar_arm].pop()
                ar_prior[ar_arm][0] += data_rew
                ar_prior[ar_arm][1] += 1 - data_rew
            else:
                realMove = True
                
        #use Hedge weights to choose arm
        sum = ar_weight + fs_weight
        true_arm = np.random.choice([ar_arm, fs_arm], p = [ar_weight/sum, fs_weight/sum])
        rew = np.random.binomial(1, arms[true_arm])
        rewards.append(rew)

        #approximate inverse propensity scores and update weights
        fs_w = approxProbs(fs_prior)
        ar_w = AR_approxProbs(ar_prior)
        EPS = 1
        ar_weight *= np.exp(EPS * (rew / ar_w[true_arm]))
        fs_weight *= np.exp(EPS * (rew / fs_w[true_arm]))

        #update TS priors
        ar_prior[true_arm][0] += rew
        ar_prior[true_arm][1] += 1 - rew

        fs_prior[true_arm][0] += rew
        fs_prior[true_arm][1] += 1 - rew
    return rewards
