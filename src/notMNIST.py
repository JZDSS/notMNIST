import os
import numpy as np
import pickle
import traceback
import tensorflow as tf
# from tensorflow.examples.tutorials import mnist

from PIL import Image

IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 64
NUM_CLASSES = 10
PIXEL_DEPTH = 255
LOGDIR = '../logs'
sess = tf.Session()
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


train_data, train_labels, valid_data, valid_labels = get_data_and_labels()
train_data.astype(np.float32)
train_labels.astype(np.int64)
valid_data.astype(np.float32)
valid_labels.astype(np.int64)
train_data = (train_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH * 2.0
valid_data = (valid_data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH * 2.0
X = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNELS])


def model(data, train=False):
    with tf.variable_scope("conv1"):
        conv1_weights = tf.get_variable("weights", initializer=tf.truncated_normal(
            [5, 5, CHANNELS, 32],  # 5x5 filter, depth 32.
            stddev=0.1))
        conv1_biases = tf.get_variable("biases", initializer=tf.zeros([32]))
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME', name="conv1")
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME', name="pool1")

    with tf.variable_scope("conv2"):
        conv2_weights = tf.get_variable("weights", initializer=tf.truncated_normal(
            [5, 5, 32, 128], stddev=0.1))
        conv2_biases = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[128]))
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME', name='conv2')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME', name='pool2')

    with tf.variable_scope("fc1"):
        fc1_weights = tf.get_variable("weights", initializer=tf.truncated_normal(
            [7, 7, 128, 1024],
            stddev=0.1))
        fc1_biases = tf.get_variable("biases", initializer=tf.constant(0.1, shape=[1024]))
        fc = tf.nn.conv2d(pool, fc1_weights, [1, 1, 1, 1], 'VALID', name='fc1')
        hidden = tf.nn.relu(fc + fc1_biases)
    
    # if train:
    #     hidden = tf.nn.dropout(hidden, 0.5)

    with tf.variable_scope("fc2"):
        fc2_weights = tf.get_variable("weights", initializer=tf.truncated_normal([1, 1, 1024, NUM_CLASSES], stddev=0.1, ))
        fc2_biases = tf.get_variable("biases", initializer=tf.constant(
            0.1, shape=[NUM_CLASSES]))
        res = tf.reshape(tf.nn.conv2d(hidden, fc2_weights, [1, 1, 1, 1], 'VALID') + fc2_biases, [-1, NUM_CLASSES])
    if train:
        tf.summary.histogram("conv1_w", conv1_weights)
        tf.summary.histogram('conv1_b', conv1_biases)
        tf.summary.histogram("conv2_w", conv2_weights)
        tf.summary.histogram("conv2_b", conv2_biases)
        tf.summary.histogram("fc1_w", fc1_weights)
        tf.summary.histogram("fc1_b", fc1_biases)
        tf.summary.histogram("fc2_w", fc2_weights)
        tf.summary.histogram("fc2_b", fc2_biases)

    return res

with tf.variable_scope("net") as scope:
    Y = model(X, True)
    scope.reuse_variables()
    P = model(X, False)

Y_ = tf.placeholder(tf.int64, [None, ])

# cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y, labels=Y_), name="loss")

cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=P, labels=Y_),name="xent")
is_correct = tf.equal(tf.argmax(tf.nn.softmax(P), 1), Y_)

accuracy1 = tf.reduce_mean(tf.cast(is_correct, tf.float32))
accuracy2 = tf.reduce_mean(tf.cast(is_correct, tf.float32))
tf.summary.scalar('acc_train', accuracy1)


batch = tf.Variable(0, dtype=tf.float32)
learning_rate = tf.train.exponential_decay(0.02, batch, 100, 0.95, staircase=True)
# tf.summary.scalar('lr', learning_rate)
# learning_rate = tf.train.exponential_decay(0.01, batch * BATCH_SIZE, 1, 0.95, staircase=True)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess.run(init)
summ = tf.summary.merge_all()
test_acc = tf.summary.scalar('acc_test', accuracy2)
saver = tf.train.Saver()
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(sess.graph)

def get_batch(data, labels):
    id = np.random.randint(low=0, high=train_labels.shape[0], size=BATCH_SIZE, dtype=np.int32)
    return data[id, ...], labels[id]


for i in range(100000):
    # get batch
    batch_X, batch_Y = get_batch(train_data, train_labels)
    
    # train
    train_feed = {X: batch_X, Y_: batch_Y}
    sess.run(train_step, feed_dict=train_feed)
    # print(sess.run(conv1_weights))
    a, c, s = sess.run([accuracy1, cross_entropy, summ], feed_dict=train_feed)
    writer.add_summary(s, i)
    print('step %06d batch acc: %02d%%, ce: %02.02f' % (i, a * 100, c))
    saver.save(sess, os.path.join(LOGDIR, 'model.ckpt'), i)
    if ((i - 99) % 100) == 0:
        valid_feed = {X: valid_data, Y_: valid_labels}
        a, c, te_a= sess.run([accuracy2, cross_entropy, test_acc], feed_dict=valid_feed)
        writer.add_summary(te_a, i/100)
        print('valid set acc: %02.02f%%, ce: %02.01f' % (a * 100, c))

