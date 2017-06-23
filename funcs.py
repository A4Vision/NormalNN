import itertools
import numpy as np
import scipy.stats

def logit(p):
    """Sigmoid inverse"""
    return - np.log(1. / p - 1.)


C = (2 * np.pi) ** 0.5
def phi(z):
    return np.exp(-0.5 * z ** 2)


def density(x, mu, sigma):
    return phi((x - mu) / sigma) / sigma / C


def relu(x):
    return np.max([x, np.zeros_like(x)], axis=0)


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def soft_exponential(x, alpha):
    return - np.log(1 - alpha * (alpha + x)) / alpha


def soft_plus(x):
    return np.log(1 + np.exp(x))


def soft_sign(x):
    return x / (1 + np.abs(x))


E_SIGMOID_NORM_COEFS1 = np.array([0, 0.04734134, - 0.18926817, 0.00844676, 1])
E_SIGMOID_NORM_COEFS2 = np.array([0, 0.0152449, - 0.08117113, - 0.072437, 1])
E_SIGMOID_NORM_COEFS3 = np.array([-0.0061311, 0.06334666, -0.20172224, 0.03032366, 1])
E_SIGMOID_NORM_COEFS4 = np.array([-0.00621259, 0.06377996, -0.20193674, 0.0289787, 1])
E_SIGMOID_NORM_COEFS5 = np.array([-0.00562129, 0.06030878, -0.20019248, 0.04200732, 1])
E_SIGMOID_NORM_COEFS6 = np.array([-0.00607378, 0.06299152, -0.20126915, 0.03073419, 1])
E_SIGMOID_NORM_COEFS7 = np.array([0.9962575559928296, -0.10864057119758552, 0.008295851536273997, -0.00033079445964674724, 5.0691849599179845e-06])


l = [E_SIGMOID_NORM_COEFS7]


def expectation_of_sigmoid_norm(mu, sigma, i=0):
    var = sigma ** 2
    A = np.array([var ** j for j in xrange(len(l[i]))])
    x = l[i]
    val = np.matmul(x, A)
    return sigmoid(val * mu)


def logit_norm_density(x, mu, sigma):
    return scipy.stats.norm.pdf(logit(x), mu, sigma) / (x * (1. - x))


def integrate(f, a, b, N):
    x = np.linspace(a + (b - a) / (2. * N), b - (b - a) / (2. * N), N)
    fx = f(x)
    area = np.sum(fx) * (b - a) / N
    return area


import cachetools
LOGIT_CACHE = cachetools.LRUCache(maxsize=2 ** 18)
LOGIT_FNAME = "logit_moments_values.pkl"


@cachetools.cached(LOGIT_CACHE)
def logit_norm_moments(mu, sigma, N=2 ** 14):
    m1 = lambda x: x * logit_norm_density(x, mu, sigma)
    avg = integrate(m1, 0.0000001, 0.9999999, N)
    m2 = lambda x: (x - avg) ** 2 * logit_norm_density(x, mu, sigma)
    var = integrate(m2, 0.0000001, 0.9999999, N)
    if avg > 1. or avg < 0:
        x = sigmoid(np.random.normal(mu, sigma, size=N / 10))
        return np.average(x), np.var(x)
    return avg, var


V_SIGMOID_NORM_COEFS = np.array([0.0006960374028368797, 0.0372228045199745, -0.00459779869023828, 0.00029805017763470497, -7.426074263403249e-06, -8.829908424968786e-06, -0.00010366042073370874, 2.624341286457866e-05, -2.286086245169167e-06, 6.477665602973575e-08, -0.00013807498480301604, -0.00383331422456413, 0.00070010404147555, -5.145696332727322e-05, 1.3489166758630117e-06, 5.646756259108541e-07, 6.750318733215978e-06, -1.7106810964508473e-06, 1.4841578444447102e-07, -4.125044818312912e-09, 4.0739641953952985e-06, 8.490129193905047e-05, -1.75931140801108e-05, 1.3645207498797505e-06, -3.671636165936132e-08])


def variance_of_sigmoid_norm(mu, sigma):
    degree_x = 5
    degree_y = 5
    assert len(V_SIGMOID_NORM_COEFS) == degree_x * degree_y
    x = mu
    y = sigma ** 2
    x_powers = [x ** i for i in xrange(degree_x)]
    y_powers = [y ** i for i in xrange(degree_y)]
    inputs = [x_powers[i] * y_powers[j] for i, j in itertools.product(range(degree_x), range(degree_y))]
    coefs = V_SIGMOID_NORM_COEFS
    return np.dot(coefs, np.array(inputs))

import utils
import tqdm

def precalc_logit_moments(mus, sigmas):
    keys = list(itertools.product(mus, sigmas))
    old = utils.get_values(keys, LOGIT_FNAME)
    for mu, sigma in tqdm.tqdm(keys):
        if (mu, sigma) not in old:
            avg, var = logit_norm_moments(mu, sigma)
            old[(mu, sigma)] = (avg, var)
    utils.dump_values(old, LOGIT_FNAME)
    LOGIT_CACHE.update(old)
    return old


if __name__ == '__main__':
    mus = np.arange(-6, 6, 0.2)
    sigmas = np.sort(np.arange(0.01, 7., 0.03))
    precalc_logit_moments(mus, sigmas)
    x, y, z1, z2 = [], [], [], []
    for mu, sigma in tqdm.tqdm(list(itertools.product(mus, sigmas))):
        avg, var = logit_norm_moments(mu, sigma)
        x.append(mu)
        y.append(sigma)
        z1.append(avg)
        z2.append(var)
    utils.plot_3d(x, y, z1, None)
    utils.plot_3d(x, y, z2, None)