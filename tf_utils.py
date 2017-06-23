import numpy as np
import tensorflow as tf


def tesnor_2d_polynomial(x, y, c):
    res = tf.constant(0, tf.float32)
    x_cur = tf.constant(1, tf.float32)
    x_degree = len(c)
    y_degree = len(c[0])
    for i in xrange(x_degree):
        y_cur = tf.constant(1, tf.float32)
        for j in xrange(y_degree):
            res += c[i][j] * x_cur * y_cur
            y_cur *= y
        x_cur *= x
    return res


def tesnor_1d_polynomial(tensor, polynomial_coefs):
    x = tensor
    f = tf.constant(polynomial_coefs[0], dtype=tensor.dtype)
    for coef in polynomial_coefs[1:]:
        f += coef * x
        x *= tensor
    return f


def test_polynomial():
    sess = tf.Session()
    x = tf.Variable(np.array([[1, 2, 3], [4, 5, 6.], [7, 8, 9.]]), dtype=tf.float32)
    init = tf.global_variables_initializer()
    poly = tesnor_1d_polynomial(x, [1., 2., 3.])
    sess.run(init)

    m = sess.run(poly)
    # print 'x'
    # print x
    # print 'm poly'
    # print m
    assert m[0, 0] == 6
    assert m[0, 1] == 1 + 2 * 2 ** 1 + 3 * 2 ** 2

# E_SIGMOID_NORM_COEFS[0] should be 0.9962575559928296, but I put 1 to simplify
# calculations over specific inputs - that is, with variance = 0
E_SIGMOID_NORM_COEFS = np.array([1, -0.10864057119758552, 0.008295851536273997, -0.00033079445964674724, 5.0691849599179845e-06])
V_SIGMOID_NORM_COEFS = np.array([[0.00069603740283687968, 0.037222804519974499, -0.0045977986902382799,
                                  0.00029805017763470497, -7.4260742634032489e-06],
                                 [-8.8299084249687863e-06, -0.00010366042073370874, 2.6243412864578658e-05,
                                  -2.2860862451691669e-06, 6.4776656029735748e-08],
                                 [-0.00013807498480301604, -0.0038333142245641299, 0.00070010404147555,
                                  -5.145696332727322e-05, 1.3489166758630117e-06],
                                 [5.6467562591085408e-07, 6.7503187332159778e-06, -1.7106810964508473e-06,
                                  1.4841578444447102e-07, -4.1250448183129118e-09],
                                 [4.0739641953952985e-06, 8.4901291939050468e-05, -1.7593114080110802e-05,
                                  1.3645207498797505e-06, -3.6716361659361318e-08]])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1. / shape[1] ** 0.5)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)


class FullyConnectedLayer(object):
    def __init__(self, input_normal, layer_size):
        assert isinstance(input_normal, TensorNormalDistribution)
        self._w = weight_variable([int(input_normal.shape[-1]), layer_size])
        self._b = bias_variable([layer_size])

        self._output_layer = input_normal.affine_transformation(self._w, self._b).sigmoid()

    def size(self):
        return self._b.shape[0]

    def norm(self, ord='euclidean'):
        return tf.norm(self._w, ord)

    def output(self):
        return self._output_layer


class SoftmaxLayer(object):
    def __init__(self, input_normal, layer_size):

        assert isinstance(input_normal, TensorNormalDistribution)
        self._w = weight_variable([int(input_normal.shape[-1]), layer_size])
        self._b = bias_variable([layer_size])

        self._after_affine = input_normal.affine_transformation(self._w, self._b)
        self._probs = self._calc_probs()

    def _calc_probs(self):
        # Side-Note: When layer_size == 2, one can implement better approximation,
        # based on the E(sigmoid(Normal)), however if this was the case then one would simply use
        # a last layer that is a neuron with sigmoid - and NOT a Softmax layer.

        # Softmax(independent normal variables)
        #   Prob(the i'th class is chosen) = E[exp(X_i) / SUM_j(exp(X_j))]
        X = self._after_affine
        probs = [None] * self.size()
        for i in xrange(self.size()):
            # p_i = 1 / SUM_j(exp(X_j - X_i))
            log_normals = []
            for j in xrange(self.size()):
                if i == j:
                    continue
                else:
                    # Subtract normals
                    normal = X[j] - X[i]
                    log_normals.append(TensorLogNormalDistribution(normal))
            # print [ln.averages().shape for ln in log_normals]
            # print [ln.averages().dtype for ln in log_normals]
            # X_i - X_i is always 0 !
            # exp(N(0, 0)) = 1 always.
            # Would use add_n, but it doesn't work since the first dimension is unknown.
            # In principle - one should have done all this computation matrix-wise instead
            # and then this problem wouldn't arise.
            average = sum([ln.averages() for ln in log_normals], 1)
            variance = sum([ln.variances() for ln in log_normals], 0)
            denominator = TensorLogNormalDistribution.from_average_and_variance(average, variance)
            probs[i] = p_i = denominator.inverse().averages()
        return tf.stack(probs, axis=1)

    def size(self):
        return self._b.shape[0]

    def norm(self, ord='euclidean'):
        return tf.norm(self._w, ord)

    def probs(self):
        return self._probs

    def cross_entropy(self, labels):
        assert str(labels.shape) == str(self._probs.shape)
        return tf.reduce_sum(labels * tf.log(self._probs))


class TensorNormalDistribution(object):
    def __init__(self, mus, variances):
        self._mus = mus
        self._var = variances

    def __add__(self, other):
        return TensorNormalDistribution(self.mus() + other.mus(), self.variances() + other.variances())

    def __sub__(self, other):
        return TensorNormalDistribution(self.mus() - other.mus(), self.variances() + other.variances())

    def mus(self):
        return self._mus

    def variances(self):
        return self._var

    def affine_transformation(self, w, b):
        # TODO: Consider covariances as well.
        return TensorNormalDistribution(tf.matmul(self._mus, w) + b,
                                        tf.matmul(self._var, w))

    def sigmoid(self):
        mu_polynomial = tesnor_1d_polynomial(self._var, E_SIGMOID_NORM_COEFS)
        hidden_mus = tf.nn.sigmoid(mu_polynomial * self._mus)

        hidden_variances = tesnor_2d_polynomial(self._mus, self._var,
                                                V_SIGMOID_NORM_COEFS)
        return TensorNormalDistribution(hidden_mus, hidden_variances)

    def __neg__(self):
        return TensorNormalDistribution(-self._mus, self._var)

    @property
    def shape(self):
        return self._mus.shape

    def __getitem__(self, i):
        return TensorNormalDistribution(self._mus[:,i], self._var[:,i])


class TensorLogNormalDistribution(object):
    def __init__(self, tensor_normal):
        assert isinstance(tensor_normal, TensorNormalDistribution)
        self._normal = tensor_normal
        self._averages = self._calc_averages()
        self._variances = self._calc_variances()

    @property
    def shape(self):
        return self._normal.shape

    def _calc_averages(self):
        return tf.exp(self._normal.mus() + self._normal.variances() / 2.0)

    def _calc_variances(self):
        return (tf.exp(self._normal.variances()) - 1.0) * self.averages() ** 2

    def averages(self):
        return self._averages

    def variances(self):
        return self._variances

    def inverse(self):
        return TensorLogNormalDistribution(-self._normal)

    @staticmethod
    def from_average_and_variance(new_average, new_variance):
        sigma_square = tf.log(new_variance / new_average ** 2 + 1)
        mu = tf.log(new_average - sigma_square / 2.)
        return TensorLogNormalDistribution(TensorNormalDistribution(mu, sigma_square))


if __name__ == '__main__':
    test_polynomial()
