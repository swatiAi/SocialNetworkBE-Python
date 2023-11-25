import tensorflow as tf
from functools import reduce
#from tensorflow.examples.tutorials.mnist import input_data

##########################
### DATASET
##########################

mnist = tf.keras.datasets.mnist

##########################
### SETTINGS
##########################

# Hyperparameters
learning_rate = 0.1
dropout_keep_proba = 0.5
epochs = 3
batch_size = 32

# Architecture
input_size = 784
image_width, image_height = 28, 28
n_classes = 10

# Other
print_interval = 500
random_seed = 123


##########################
### WRAPPER FUNCTIONS
##########################

def conv2d(input_tensor, output_channels,
           kernel_size=(5, 5), strides=(1, 1, 1, 1),
           padding='SAME', activation=None, seed=None,
           name='conv2d'):
    with tf.name_scope(name):
        input_channels = input_tensor.get_shape().as_list()[-1]
        weights_shape = (kernel_size[0], kernel_size[1],
                         input_channels, output_channels)

        weights = tf.Variable(tf.truncated_normal(shape=weights_shape,
                                                  mean=0.0,
                                                  stddev=0.01,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=(output_channels,)), name='biases')
        conv = tf.nn.conv2d(input=input_tensor,
                            filter=weights,
                            strides=strides,
                            padding=padding)

        act = conv + biases
        if activation is not None:
            act = activation(conv + biases)
        return act


def fully_connected(input_tensor, output_nodes,
                    activation=None, seed=None,
                    name='fully_connected'):
    with tf.name_scope(name):
        input_nodes = input_tensor.get_shape().as_list()[1]
        weights = tf.Variable(tf.truncated_normal(shape=(input_nodes,
                                                         output_nodes),
                                                  mean=0.0,
                                                  stddev=0.01,
                                                  dtype=tf.float32,
                                                  seed=seed),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[output_nodes]), name='biases')

        act = tf.matmul(input_tensor, weights) + biases
        if activation is not None:
            act = activation(act)
        return act


##########################
### GRAPH DEFINITION
##########################

g = tf.Graph()
with g.as_default():
    tf.set_random_seed(random_seed)

    # Input data
    tf_x = tf.placeholder(tf.float32, [None, input_size, 1], name='inputs')
    tf_y = tf.placeholder(tf.float32, [None, n_classes], name='targets')

    keep_proba = tf.placeholder(tf.float32, shape=None, name='keep_proba')

    # Convolutional Neural Network:
    # 2 convolutional layers with maxpool and ReLU activation
    input_layer = tf.reshape(tf_x, shape=[-1, image_width, image_height, 1])

    conv1 = conv2d(input_tensor=input_layer,
                   output_channels=8,
                   kernel_size=(3, 3),
                   strides=(1, 1, 1, 1),
                   activation=tf.nn.relu,
                   name='conv1')

    pool1 = tf.nn.max_pool(conv1,
                           ksize=(1, 2, 2, 1),
                           strides=(1, 2, 2, 1),
                           padding='SAME',
                           name='maxpool1')

    conv2 = conv2d(input_tensor=pool1,
                   output_channels=16,
                   kernel_size=(3, 3),
                   strides=(1, 1, 1, 1),
                   activation=tf.nn.relu,
                   name='conv2')

    pool2 = tf.nn.max_pool(conv2,
                           ksize=(1, 2, 2, 1),
                           strides=(1, 2, 2, 1),
                           padding='SAME',
                           name='maxpool2')

    dims = pool2.get_shape().as_list()[1:]
    dims = reduce(lambda x, y: x * y, dims, 1)
    flat = tf.reshape(pool2, shape=(-1, dims))

    out_layer = fully_connected(flat, n_classes, activation=None,
                                name='logits')

    # Loss and optimizer
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=out_layer, labels=tf_y)
    cost = tf.reduce_mean(loss, name='cost')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(cost, name='train')

    # Prediction
    correct_prediction = tf.equal(tf.argmax(tf_y, 1),
                                  tf.argmax(out_layer, 1),
                                  name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32),
                              name='accuracy')

import numpy as np

##########################
### TRAINING & EVALUATION
##########################

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    np.random.seed(random_seed)  # random seed for mnist iterator
    for epoch in range(1, epochs + 1):
        avg_cost = 0.
        total_batch = mnist.train.num_examples // batch_size

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x[:, :, None]  # add "missing" color channel

            _, c = sess.run(['train', 'cost:0'],
                            feed_dict={'inputs:0': batch_x,
                                       'targets:0': batch_y,
                                       'keep_proba:0': dropout_keep_proba})
            avg_cost += c
            if not i % print_interval:
                print("Minibatch: %03d | Cost: %.3f" % (i + 1, c))

        train_acc = sess.run('accuracy:0',
                             feed_dict={'inputs:0': mnist.train.images[:, :, None],
                                        'targets:0': mnist.train.labels,
                                        'keep_proba:0': 1.0})
        valid_acc = sess.run('accuracy:0',
                             feed_dict={'inputs:0': mnist.validation.images[:, :, None],
                                        'targets:0': mnist.validation.labels,
                                        'keep_proba:0': 1.0})

        print("Epoch: %03d | AvgCost: %.3f" % (epoch, avg_cost / (i + 1)), end="")
        print(" | Train/Valid ACC: %.3f/%.3f" % (train_acc, valid_acc))

    test_acc = sess.run('accuracy:0',
                        feed_dict={'inputs:0': mnist.test.images[:, :, None],
                                   'targets:0': mnist.test.labels,
                                   'keep_proba:0': 1.0})

    print('Test ACC: %.3f' % test_acc)
