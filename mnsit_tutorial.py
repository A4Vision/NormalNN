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


W = weight_variable([784, 10])
b = bias_variable([10])

#
# sess.run(tf.global_variables_initializer())
#
# y = tf.matmul(x, W) + b
#
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
# )
#
#
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train_step = optimizer.minimize(cross_entropy)
#
#
# correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
#
# t = time.time()
# for i in xrange(10000):
#     batch = mnist.train.next_batch(100)
#     train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#     if i % 1000 == 0:
#         print 'accuracy', i, accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
# print time.time() - t


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

x_image = tf.reshape(x, [-1,28,28,1])

N1, N2 = 10, 5

with tf.name_scope('hidden1'):
    W_conv1 = weight_variable([5, 5, 1, N1])
    b_conv1 = bias_variable([N1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('hidden2'):
    W_conv2 = weight_variable([5, 5, N1, N2])
    b_conv2 = bias_variable([N2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * N2, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*N2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name="CrossEntropy"
)

tf.summary.scalar('loss', cross_entropy)
global_step = tf.Variable(0, trainable=False)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)
sess.run(tf.global_variables_initializer())
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("train_folder", sess.graph)

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

for i in tqdm.tqdm(range(20000)):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

        if i % 500 == 0:
            saver.save(sess, "train_folder/output.ckpt", global_step=global_step)
            feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})



