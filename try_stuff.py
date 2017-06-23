import numpy as np
import matplotlib.pyplot as plt
import funcs




N = 2 ** 17


def normal_through_func(m, s, func, p):
    x = np.random.normal(m, s, size=N)
    z = func(x)
    print func.func_name, min(z), max(z)
    p.hist(z, bins=100, normed=True)
    a = np.arange(min(z), max(z), step=0.001)
    z_normal = funcs.density(a, np.average(z), np.var(z) ** 0.5)
    p.plot(a, z_normal, linewidth=3)


def plot_func(func):
    x = np.arange(-2, 2, 0.001)
    plt.plot(x, func(x))

soft_exp_good = lambda x: funcs.soft_exponential(x, 0.35)

l = (funcs.sigmoid, funcs.soft_sign, funcs.soft_plus, soft_exp_good)
mus = np.arange(-2, 3.1, 1.)

for j, mu in enumerate(mus):
    # plt.hist2d(x, y, bins=40, norm=LogNorm())
    plt.colorbar()
    plt.show()
    fig, axarr = plt.subplots(len(l))
    for p, f in zip(axarr, l):
        p.set_title("mu={}, func={}".format(mu, f.func_name))
        normal_through_func(mu, 0.05, f, p)
    plt.show()

