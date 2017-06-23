import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1),
            tf.contrib.layers.real_valued_column("abc", dimension=2),
            tf.contrib.layers.real_valued_column("abcd", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
linear_regression = tf.contrib.learn.LinearRegressor(feature_columns=features)

# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
    # Build a linear model and predict values
    W = tf.get_variable("W", [1], dtype=tf.float64)
    W2 = tf.get_variable("W2", [2], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)

    z = W2 * features["abc"]
    s = tf.reduce_sum(z)

    y = W * features['x'] + W2 * features['abc'][0] + b

    # Loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))

    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.02)
    train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))

    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

custom_estimator = tf.contrib.learn.Estimator(model_fn=model)

for estimator in (linear_regression, custom_estimator):
    # TensorFlow provides many helper methods to read and set up data sets.
    # Here we use `numpy_input_fn`. We have to tell the function how many batches
    # of data (num_epochs) we want and how big each batch should be.
    x = np.array([1., 2., 3., 4.])
    abc = np.array([[1., 1], [2, 2], [3, 3.2], [4, 4]], dtype=np.float64)
    abcd = np.arange(0, 4, 1, dtype=np.float64)
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"abc": abc, "x": x, "abcd": abcd}, y, batch_size=2, num_epochs=100)

    # We can invoke 1000 training steps by invoking the `fit` method and passing the
    # training data set.

    estimator.fit(input_fn=input_fn, steps=100)
    print estimator
    print(estimator.evaluate(input_fn=input_fn))


