import argparse
import sys
import os
import time
import numpy as np
import pickle
import traceback
import tensorflow as tf
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None

IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 128
NUM_CLASSES = 10
PIXEL_DEPTH = 255


def get_data_and_labels():
    if not os.path.exists('train_data.pkl'):
        letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        train_num = 0
        valid_num = 0
        for letter in letter_list:
            im_dir = os.listdir('../data/notMNIST_large/%s' % letter)
            train_num += len(im_dir)
            m_dir = os.listdir('../data/notMNIST_small/%s' % letter)
            valid_num += len(im_dir)

        train_data = np.ndarray(shape=[train_num, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        rain_labels = np.ndarray(shape=[train_num])
        valid_data = np.ndarray(shape=[valid_num, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
        valid_labels = np.ndarray(shape=[valid_num])

        def load_data(dir, num):
            n = -1
            label = -1
            data = np.ndarray(shape=[num, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
            labels = np.ndarray(shape=[num])
            for letter in letter_list:
                label += 1
                im_dir = os.listdir('%s/%s' % (dir, letter))
                for image_name in im_dir:
                    try:
                        image = Image.open('%s/%s/%s' % (dir, letter, image_name))
                    except:
                        f = open("c:log.txt", 'a')
                        traceback.print_exc(file=f)
                        f.flush()
                        f.close()
                        continue
                    n += 1
                    data[n, ...] = np.array(image).reshape([IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
                    labels[n] = label
                    print('%s/%s/%s' % (dir, letter, image_name))
            return data, labels

        train_data, train_labels = load_data('../data/notMNIST_large', train_num)
        valid_data, valid_labels = load_data('../data/notMNIST_small', valid_num)

        pickle.dump(train_data, open('train_data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_labels, open('train_labels.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_data, open('valid_data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_labels, open('valid_labels.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        train_data = pickle.load(open('train_data.pkl', 'rb'))
        train_labels = pickle.load(open('train_labels.pkl', 'rb'))
        valid_data = pickle.load(open('valid_data.pkl', 'rb'))
        valid_labels = pickle.load(open('valid_labels.pkl', 'rb'))
    return train_data, train_labels, valid_data, valid_labels


def deepnn(x):
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="weights")

    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="biases")

    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            with tf.name_scope(name):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32])
        variable_summaries(W_conv1, 'weights')
        b_conv1 = bias_variable([32])
        variable_summaries(b_conv1, 'biases')

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64])
        variable_summaries(W_conv2, 'weights')
        b_conv2 = bias_variable([64])
        variable_summaries(b_conv2, 'biases')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        variable_summaries(W_fc1, 'weights')
        b_fc1 = bias_variable([1024])
        variable_summaries(b_fc1, 'biases')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([1024, 10])
        variable_summaries(W_fc2, 'weights')
        b_fc2 = bias_variable([10])
        variable_summaries(b_fc2, 'biases')

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def train():
    # train_data, train_labels, valid_data, valid_labels = get_data_and_labels()
    # train_data.astype(np.float32)
    # train_labels.astype(np.int64)
    # valid_data.astype(np.float32)
    # valid_labels.astype(np.int64)
    # train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH * 2.0
    # valid_data = (valid_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH * 2.0
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)


    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.int64, [None, 10], name='y-input')
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])

    # with tf.name_scope('input_reshape'):
    #     image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

    y, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y), name="cross_entropy")
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        global_step = tf.Variable(0, name="global_step")
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                                   global_step, FLAGS.decay_steps, 0.95, True, "learning_rate")
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(
            cross_entropy, global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()

    with tf.name_scope("saver"):
        saver = tf.train.Saver(name="saver")

    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

    if tf.gfile.Exists(os.path.join(FLAGS.ckpt_dir, 'checkpoint')):
        saver.restore(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
        # acc = sess.run(accuracy, feed_dict={x: train_data[1:5000, ...], y_: train_labels[1:5000], keep_prob: 1.0})
        # print(acc)
    else:
        tf.global_variables_initializer().run()

    def feed_dict(train, kk=FLAGS.dropout):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = kk
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    for i in range(FLAGS.max_steps + 1):
        if i % 1000 == 0 and i != 0:
            time.sleep(100)

        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))

        # else:  # Record train set summaries, and train
        if i % 10 == 9:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            feed = feed_dict(True)
            sess.run(train_step,
                                  feed_dict=feed,
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            feed[keep_prob] = 1.0
            summary = sess.run(merged, feed_dict=feed)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for step', i)
            saver.save(sess, os.path.join(FLAGS.ckpt_dir, 'model.ckpt'))
        else:  # Record a summary
            feed = feed_dict(True)
            sess.run(train_step, feed_dict=feed)
            feed[keep_prob] = 1.0
            summary = sess.run(merged, feed_dict=feed)
            train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                        help='Initial learning rate')
    parser.add_argument('--decay_steps', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.',
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../logs',
        help='Summaries log directory')
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='../ckpt',
        help='Check point directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
