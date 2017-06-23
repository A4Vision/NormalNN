import numpy as np
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
sess = tf.Session()
print sess.run([node1, node2])
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
sess.close()
sess = tf.Session()
print "run(adder_node)", sess.run(adder_node, feed_dict={a: [1, 123], b: [213, 123]})
triple_3 = adder_node * 3
print "run(triple_3)", sess.run(triple_3, feed_dict={a: 3, b: 1})

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)
print 'sess.run(linear_model, {x: [1, 2, 3, 4]})', sess.run(linear_model, {x: [1, 2, 3, 4]})

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1,2,3,4], y: [0,-1,-2,-3]}))


fixW = tf.assign(W, [4.2])
fixb = tf.assign(b, [0.])
sess.run([fixW, fixb])
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
for i in range(6):
    print sess.run([W, b, loss], {x:[1,2,3,4], y:[0,-1,-2,-3]})
    sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})


W = tf.Variable([1, 3.], tf.float32)
x = tf.placeholder(tf.float32, shape=2)
z = W * x
s = tf.reduce_sum(z)


droped_x = tf.nn.dropout(x, 0.5)
matrix = tf.Variable(np.array([[1, 2, 3], [4, 5, 6.], [7, 8, 9.]]), dtype= tf.float32)
droped_matrix = tf.nn.dropout(matrix, 0.5)
# print 'shape', matrix.shape, droped_x.shape
init = tf.global_variables_initializer()
pow_x = tf.pow(x, 2)
sess.run(init)

mult_matrix = tf.constant(10.2) * matrix

# print 'matrix mult', sess.run(mult_matrix, {x: [1, 3]})
# print 'pow x', sess.run(pow_x, {x: [1, 3]})
pow_matrix = matrix * matrix
# print 'pow matrix', sess.run(pow_matrix)
# print 'run(s)', sess.run([s, droped_x, droped_matrix], {x: [1, 2.]})

x = tf.placeholder(tf.float32, shape=(5,))
y = tf.placeholder(tf.float32, shape=(5,))

def f(x, y):
    return x ** 2 + 3 * y * x - y ** 2 - 3
#
c = [[-3, 0, -1], [0, 3, 0], [1, 0, 0]]


def f2(x, y, c):
    res = 0
    for i in xrange(len(c)):
        for j in xrange(len(c[0])):
            res += c[i][j] * x ** i * y ** j
    return res


def f3(x, y, c):
    res = 0
    x_cur = 1

    for i in xrange(len(c)):
        y_cur = 1
        for j in xrange(len(c[0])):
            res += c[i][j] * x_cur * y_cur
            y_cur *= y
        x_cur *= x
    return res


def polynomial2d_tf(x, y, c):
    assert x.shape == y.shape
    shape = x.shape
    res = tf.constant(0, tf.float32, shape)
    x_cur = tf.constant(1, tf.float32, shape)
    x_degree = len(c)
    y_degree = len(c[0])
    for i in xrange(x_degree):
        y_cur = tf.constant(1, tf.float32, shape)
        for j in xrange(y_degree):
            res += c[i][j] * x_cur * y_cur
            y_cur *= y
        x_cur *= x
    return res


p = polynomial2d_tf(x, y, c)
q = (p / 2.) ** 2
z = p[1] + p[2]
w = tf.reduce_sum(p) - p[1]
concat1 = tf.stack([w, w, w], 0)
concat2 = tf.stack([w, w, w])
concat_val1, concat_val2, w_value, z_value, divided, poly_value = sess.run([concat1, concat2, w, z, q, p], feed_dict={x: [21.] * 5, y: [-2.] * 5})
print concat_val1, concat_val2
print f(21, -2), f2(21, -2, c), f3(21, -2, c), poly_value, divided, z_value, w_value

assert f(21, -2) == f2(21, -2, c) == f3(21, -2, c) == poly_value[0] == poly_value[1]

