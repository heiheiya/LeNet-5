import tensorflow as tf

def conv_layer(input, ksize, strides, name, b_value = 0.0, padding='VALID'):
    with tf.name_scope(name) as scope:
        filter = tf.Variable(tf.truncated_normal(ksize, dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(input=input, filter=filter, strides=strides, padding=padding)
        biases = tf.Variable(tf.constant(b_value, shape=[ksize[3]], dtype=tf.float32), trainable=True, name='biases')
        z = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(z, name=scope)
        print_activations(relu)
        return relu


def max_pool_layer(input, ksize, strides, name, padding='VALID'):
    max_pool = tf.nn.max_pool(input, ksize=ksize, strides=strides, padding=padding, name=name)
    print_activations(max_pool)
    return max_pool

def full_connected_layer(input, n_out, name, keep_prob = 1.0, b_value = 0.0):
    shape = input.get_shape().as_list()
    dim = 1
    for d in range(len(shape)-1):
        dim *= shape[d+1]
    x = tf.reshape(input, [-1, dim])
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=1e-1),name='weights')
        biases = tf.Variable(tf.constant(b_value, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        fc = tf.nn.relu(tf.add(tf.matmul(x, weights), biases), name=scope)
        fc_drop = tf.nn.dropout(fc, keep_prob)
        print_activations(fc_drop)
        return fc_drop

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())