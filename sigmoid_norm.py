import time

import itertools

print time.time()

import cPickle
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import funcs
from sklearn import linear_model
import utils


def get_samples(sigmas, mus, N):
    coefs = []
    for sigma in sigmas:
        averages = []

        for mu in mus:
            averages.append(funcs.logit_norm_moments(mu, sigma)[0])

        averages = np.array(averages)
        logit_values = funcs.logit(averages)
        X = np.array([mus, np.ones_like(mus)]).T
        Y = logit_values
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(X, Y)
        coefs.append(regr.coef_[0])
        print "score={} s={} coef={}".format(regr.score(X, Y), sigma, regr.coef_.tolist())
        # plt.plot(mus, logit_values, '>')
        # plt.plot(mus, mus * regr.coef_[0], '-*')
        # plt.show()
    return coefs


def find_expectation_of_sigmoid(mus, sigmas, N):
    coefs = get_samples(sigmas, mus, N)
    variances = sigmas ** 2
    X = np.stack([variances ** i for i in xrange(5)]).T
    Y = coefs
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(X, Y)

    print "score={} coef={}".format(regr.score(X, Y), regr.coef_.tolist())
    Y2 = np.array((np.matrix(regr.coef_) * np.matrix(X).T))[0]
    plt.plot(sigmas, Y2, '-')
    plt.plot(sigmas, Y, '*')
    plt.show()


def calc_errors(mu, sigma, N):
    global good, bad
    x = np.random.normal(mu, sigma, size=N)
    z = funcs.sigmoid(x)
    avg = np.average(z)
    expected = [funcs.expectation_of_sigmoid_norm(mu, sigma, i) for i in xrange(len(funcs.l))]
    return [np.abs(e - avg) / (np.abs(e) + np.abs(avg)) for e in expected]


def main_verify_expectancy(mus, sigmas, N):
    print time.time()
    pairs = list(itertools.product(mus, sigmas))
    mus_flat = np.array([x[0] for x in pairs])
    sigmas_flat = np.array([x[1] for x in pairs])
    estimated_averages = funcs.expectation_of_sigmoid_norm(mus_flat, sigmas_flat)
    actual_averages = np.array([funcs.logit_norm_moments(mu, sigma)[0] for mu, sigma in pairs])
    utils.plot_3d(mus_flat, sigmas_flat, estimated_averages, actual_averages)
    utils.plot_3d(mus_flat, sigmas_flat, np.abs(estimated_averages - actual_averages) / (estimated_averages + actual_averages), None)


def calc_error(real, wanted):
    error = np.abs(real - wanted) / np.abs(wanted)
    return error


def main_verify_variance(points):
    # Note: When exceeding the interpolation range, (mu < -4 for instance)
    #       then the error is significant.
    errors_points = []
    for mu, variance, actual in points:
        estimate = funcs.variance_of_sigmoid_norm(mu, variance ** 0.5)
        error = calc_error(actual, estimate)
        errors_points.append((mu, variance, error, actual, estimate))
    x, y, relative_error, actual, estimate = np.array(errors_points).T
    utils.plot_3d(x, y, relative_error, None)
    utils.plot_3d(x, y, np.abs(actual - estimate), None)


def main_expectancy():
    mus = np.arange(-6, 6, 0.2)
    sigmas = np.sort(np.arange(0.01, 5., 0.03))
    funcs.precalc_logit_moments(mus, sigmas)
    find_expectation_of_sigmoid(mus, sigmas, 2 ** 16)
    main_verify_expectancy(mus, sigmas, 2 ** 17)


def variance_points(mus, sigmas):
    return [(x, y ** 2, w) for x, y, z, w in gen_points(mus, sigmas)]


def main_variance():
    mus = np.arange(-6, 6, 0.1)
    sigmas = np.sort(np.concatenate((np.arange(0.05, 0.2, 0.05), np.arange(0.01, 4., 0.1))))
    funcs.precalc_logit_moments(mus, sigmas)
    points = variance_points(mus, sigmas)
    # coefs = interpolate(points, 5, 5, True)
    # print 'coefs'
    # print coefs.tolist()
    main_verify_variance(points)


def interpolate(points, degree_x, degree_y, plot=False):
    x, y, z = np.array(points).T
    x_powers = [x ** i for i in xrange(degree_x)]
    y_powers = [y ** i for i in xrange(degree_y)]
    inputs = [x_powers[i] * y_powers[j] for i, j in itertools.product(range(degree_x), range(degree_y))]
    regr = linear_model.LinearRegression(fit_intercept=False, n_jobs=2)
    regr.fit(np.array(inputs).T, z)
    coefs = regr.coef_
    z_predicted = np.dot(coefs, np.array(inputs))
    if plot:
        utils.plot_3d(x, y, z, z_predicted)
        utils.plot_3d(x, y, z - z_predicted, None)
    return coefs


def gen_points(mus, sigmas):
    points = []
    for mu in mus:
        for sigma in sigmas:
            z = tuple(funcs.logit_norm_moments(mu, sigma))
            points.append((mu, sigma) + z)
    return points


if __name__ == '__main__':
    # main_expectancy()
    main_variance()
