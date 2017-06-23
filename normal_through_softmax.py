import itertools
import random
import numpy as np
import funcs
from math import e
from scipy import integrate
from scipy import stats
import collections
random.seed(12345)
np.random.seed(1234)

def integrate2d(func, x1, x2, y1, y2, args, N=2 ** 15):
    N1 = N2 = int(N ** 0.5)
    x, y = np.meshgrid(np.linspace(x1, x2, N1),
                       np.linspace(y1, y2, N2))
    z = func(x, y, *args)
    small = (x2 - x1) / float(N1) * (y2 - y1) / float(N2)
    return z.sum() * small


def bernouli_through_softmax_density(x, y, mu1, mu2, sigma1, sigma2):
    return e ** x / (e ** x + e ** y) * stats.norm.pdf(x, mu1, sigma1) * stats.norm.pdf(y, mu2, sigma2)


def test_integrate():
    for args in [(0.0, 0.0, .1, .1), (0.0, 2.0, .1, .1)]:
        print args
        print integrate.dblquad(bernouli_through_softmax_density, -10., 10., lambda x: -10., lambda x: 10.,
                                args=args, epsabs=e-4, epsrel=1e-3)
        print integrate2d(bernouli_through_softmax_density, -10, 10, -10, 10, args)


class NormalVariable(object):
    def __init__(self, mu, variance):
        self._mu = mu
        self._var = variance

    def __add__(self, other):
        return NormalVariable(self.mu() + other.mu(), self.variance() + other.variance())

    def mu(self):
        return self._mu

    def variance(self):
        return self._var

    def apply_sigmoid(self):
        mu, variance = funcs.logit_norm_moments(self._mu, self._var ** 0.5)
        return NormalVariable(mu, variance)

    def __mul__(self, other):
        return NormalVariable(self._mu * other, self._var * other ** 2)

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return  self * (-1)

    def randomize(self, size=None):
        return np.random.normal(self._mu, self._var ** 0.5, size)

    def __str__(self):
        return repr(self).replace("NormalVariable", "N")

    def __repr__(self):
        return "NormalVariable(mu={}, variance={})".format(self._mu, self._var)


class LogNormalVariable(object):
    def __init__(self, nv):
        self._normal = nv

    def average(self):
        return np.exp(self._normal.mu() + self._normal.variance() / 2.)

    def variance(self):
        return (np.exp(self._normal.variance()) - 1) * self.average() ** 2

    def inverse(self):
        return LogNormalVariable(-self._normal)

    def randomize(self, size=None):
        return np.exp(self._normal.randomize(size))

    def __str__(self):
        return "exp({})".format(self._normal)

    def __repr__(self):
        return "LogNormalVariable({})".format(repr(self._normal))

    @staticmethod
    def sum_log_normals(log_normals):
        top = sum(ln.variance() for ln in log_normals)
        bottom = sum(ln.average() for ln in log_normals) ** 2
        sum_variance = np.log(top / bottom + 1)
        sum_mu = np.log(sum(ln.average() for ln in log_normals)) - sum_variance / 2.
        return LogNormalVariable(NormalVariable(sum_mu, sum_variance))


def softmax(x):
    diff = x - np.min(x, axis=0)
    return np.exp(diff) / np.sum(np.exp(diff), axis=0)

print 1 / (1 + e ** 3)
print softmax(np.array([[1, 2, 3], [1, 5, 6]]))


class NormalVariablesVector(object):
    def __init__(self, normals):
        self._normals = list(normals)

    def apply_sigmoid(self):
        return NormalVariablesVector([n.apply_sigmoid() for n in self._normals])

    def apply_softmax(self):
        pass

    def __len__(self):
        return len(self._normals)

    def randmoize_by_softmax(self, N):
        vecs = np.array([n.randomize(N) for n in self._normals])
        probs = softmax(vecs)
        selections = [np.random.choice(range(len(self)), p=probs_vec) for probs_vec in probs.T]
        m = collections.Counter(selections)
        assert sum(m.values()) == N
        return [m[i] / float(N) for i in xrange(len(self))]

    def estimate_softmax_probs(self):
        probs = []
        for i in xrange(len(self)):
            log_normals = [LogNormalVariable(NormalVariable(0, 0))]
            for j in xrange(len(self)):
                if i == j:
                    continue
                diff = self._normals[j] - self._normals[i]
                log_normals.append(LogNormalVariable(diff))
            inverse = LogNormalVariable.sum_log_normals(log_normals).inverse()
            p_i = inverse.average()
            probs.append(p_i)
        return probs


def test_randomization():
    N = 2 ** 20
    for args in [(0.0, 0.0, .1, .1), (0.0, 2.0, .1, .1)]:
        print 'args', args
        x = np.random.normal(args[0], args[2], N)
        y = np.random.normal(args[1], args[3], N)
        p = np.exp(x) / (np.exp(x) + np.exp(y))
        prob_select0 = np.sum(np.random.random(size=N) < p) / float(len(p))
        print 'prob_select0', prob_select0

mus = np.arange(-5, 5, 0.2)
sigmas = np.sort(np.arange(0.01, 1.3, 0.03))
d = {}
for i in xrange(100):
    pairs = random.sample(list(itertools.product(mus, sigmas)), 10)
    variables = [NormalVariable(mu, sigma ** 2) for mu, sigma in pairs]
    normals = NormalVariablesVector(variables)
    sampled = normals.randmoize_by_softmax(2 ** 12)
    estimate = normals.estimate_softmax_probs()
    tv = np.average(np.abs(np.array(sampled) - np.array(estimate)))
    d[tv] = pairs
    print tv, i

for tv in d:
    if tv > 0.1:
        print tv
        print d[tv]


# print -NormalVariable(10, 2)
# x = NormalVariable(0, 1).randomize(100)
# print NormalVariable(0, 0.1).apply_sigmoid().apply_sigmoid() * -1
#
print sum([NormalVariable(1, 0), NormalVariable(2, 3)], -NormalVariable(1, 1))
