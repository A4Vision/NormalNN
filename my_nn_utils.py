import tensorflow as tf


def dropout(x1, x2, keep_prob, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
  """Computes dropout on two vectors.
  """
  with tf.ops.name_scope(name, "dropout", [x]) as name:
    x = tf.ops.convert_to_tensor(x, name="x")
    if isinstance(keep_prob, tf.numbers.Real) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = tf.ops.convert_to_tensor(keep_prob,
                                      dtype=x.dtype,
                                      name="keep_prob")
    keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

    # Do nothing if we know keep_prob == 1
    if tf.tensor_util.constant_value(keep_prob) == 1:
      return x1, x2

    noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob
    random_tensor += tf.random_ops.random_uniform(noise_shape,
                                               seed=seed,
                                               dtype=x.dtype)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.math_ops.floor(random_tensor)
    ret1 = tf.math_ops.div(x1, keep_prob) * binary_tensor
    ret2 = tf.math_ops.div(x2, keep_prob) * binary_tensor
    ret1.set_shape(x.get_shape())
    ret2.set_shape(x.get_shape())
    return ret1, ret2