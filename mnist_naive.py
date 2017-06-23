import time
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print 'loading'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print 'loaded !'
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

N1, N2, N3 = 400, 50, 50


def fc_layer(prev_layer, prev_layer_size, size, keep_prob):
    W1 = weight_variable([prev_layer_size, size])
    b1 = bias_variable([size])
    h_fc1 = tf.nn.relu(tf.matmul(prev_layer, W1) + b1)
    return tf.nn.dropout(h_fc1, keep_prob)


keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('hidden1'):
    h1 = fc_layer(x, 784, N1, keep_prob)

with tf.name_scope('hidden2'):
    h2 = fc_layer(h1, N1, N2, keep_prob)

with tf.name_scope('hidden3'):
    h3 = fc_layer(h2, N2, N3, keep_prob)

with tf.name_scope('final'):
    W1 = weight_variable([N3, 10])
    b1 = bias_variable([10])
    final = tf.nn.relu(tf.matmul(h3, W1) + b1)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final), name="CrossEntropy"
)

tf.summary.scalar('loss', cross_entropy)
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(final,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)
sess.run(tf.global_variables_initializer())
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("train_folder2", sess.graph)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

for i in tqdm.tqdm(range(50000)):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

        if i % 500 == 0:
            saver.save(sess, "train_folder2/output.ckpt", global_step=global_step)
            feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



