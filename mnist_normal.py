import numpy as np

import glob
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import tf_utils

print 'loading'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print 'loaded !'
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
x_var = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


N1, N2, N3 = 2000, 200, 100

ws = []

input_normals = tf_utils.TensorNormalDistribution(x, x_var)
l1 = tf_utils.FullyConnectedLayer(input_normals, N1)
l2 = tf_utils.FullyConnectedLayer(l1.output(), N2)
l3 = tf_utils.FullyConnectedLayer(l2.output(), N3)
l4 = tf_utils.SoftmaxLayer(l3.output(), 10)

cross_entropy = l4.cross_entropy(y_)

normalization_loss = tf.add_n([l.norm() for l in (l1, l2, l3, l4)])

lambda1 = tf.placeholder(tf.float32)

total_loss = cross_entropy + lambda1 * normalization_loss


tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('normalization', normalization_loss)
tf.summary.scalar('loss', total_loss)
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(l4.probs(), 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)
summary = tf.summary.merge_all()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
DIR = "train_folder_normal"
lambda1_value = 5e-4
PARAMS_LOCATION = DIR + "/output.ckpt"
if glob.glob(PARAMS_LOCATION + "*"):
    print glob.glob(PARAMS_LOCATION + "*")
    saver.restore(sess, PARAMS_LOCATION)
print "removing old files..."
if not os.path.exists(DIR):
    os.mkdir(DIR)
for i in os.listdir(DIR):
    os.remove(os.path.join(DIR, i))
summary_writer = tf.summary.FileWriter(DIR, sess.graph)
print 'lambda1_value', lambda1_value
for i in range(4000):
    batch = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1],
                                                  x_var: np.zeros(batch[0].shape, dtype=np.float32),
                                                  lambda1: lambda1_value})
        print("step %d, training accuracy %g" % (i, train_accuracy))

        if i % 500 == 0:
            saver.save(sess, PARAMS_LOCATION)
            feed_dict = {x: mnist.test.images, y_: mnist.test.labels, lambda1: lambda1_value,
                         x_var: np.zeros(mnist.test.images.shape, dtype=np.float32)}
            summary_str, accuracy_actual = sess.run([summary, accuracy], feed_dict=feed_dict)
            print 'test accuracy:', accuracy_actual
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

    train_step.run(feed_dict={x: batch[0], y_: batch[1],
                              x_var: np.zeros(batch[0].shape, dtype=np.float32),
                              lambda1: lambda1_value})



