import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os



print 'loading'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print 'loaded !'
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1. / shape[1] ** 0.5)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)


N1, N2, N3 = 2000, 200, 100

ws = []


def fc_layer(prev_layer, prev_layer_size, size, keep_prob_placeholder):
    W1 = weight_variable([prev_layer_size, size])
    ws.append(W1)
    b1 = bias_variable([size])
    h_fc1 = tf.nn.sigmoid(tf.matmul(prev_layer, W1) + b1)
    return h_fc1


keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('hidden1'):
    h1 = fc_layer(x, 784, N1, keep_prob)

with tf.name_scope('hidden2'):
    h2 = fc_layer(h1, N1, N2, keep_prob)

with tf.name_scope('hidden3'):
    h3 = fc_layer(h2, N2, N3, keep_prob)

with tf.name_scope('final'):
    W1 = weight_variable([N3, 10])
    ws.append(W1)
    b1 = bias_variable([10])
    final = tf.nn.sigmoid(tf.matmul(h3, W1) + b1)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final), name="CrossEntropy"
)


normalization_loss = tf.add_n([tf.norm(w) for w in ws])

lambda1 = tf.placeholder(tf.float32)

total_loss = cross_entropy + lambda1 * normalization_loss


tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('normalization', normalization_loss)
tf.summary.scalar('loss', total_loss)
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(1e-3).minimize(total_loss, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(final,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)
summary = tf.summary.merge_all()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

stime = time.time()
for lambda1_value in (4e-6,):
    sess.run(tf.global_variables_initializer())
    DIR = "train_folder2_{}".format(lambda1_value)
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
    for i in range(50000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0,
                                                      lambda1: lambda1_value})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            if i % 500 == 0:
                print time.time() - stime
                stime = time.time()
                saver.save(sess, PARAMS_LOCATION)
                feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, lambda1: lambda1_value}
                summary_str, accuracy_actual, total_loss1, normalization_loss1, cross_entropy1 = sess.run([summary, accuracy,
                                                         total_loss, normalization_loss, cross_entropy], feed_dict=feed_dict)
                print 'test accuracy:', accuracy_actual
                print total_loss1, normalization_loss1 * lambda1_value, cross_entropy1
                summary_writer.add_summary(summary_str, i)
                summary_writer.flush()

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.4,
                                  lambda1: lambda1_value})


