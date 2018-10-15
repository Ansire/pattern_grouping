"""
Element encoder implementation using Tensorflow based on the paper
Learning to Group Discrete Graphical Patterns
Author Wanchao Su
"""

import tensorflow as tf
import numpy as np
import os
import glob
import scipy.misc
import time
from random import shuffle

PHASE = 'train'

# Directory of the training/testing data, change the directory according to different phases
DATA_DIR = './dataset/'

OUTPUT_DIR = './output_4scales/'
CHECKPOINT_DIR = './checkpoint/'
LOG_DIR = './logs/'

# view sizes of the training process
VIEW_SIZE = 4

# keep rate in the dropout layer
KEEP_PROB = 1

MAX_ITER = 60009
STEP_SIZE = 100

# positive and negative data lists for training the network
POS_LIST = []
NEG_LIST = []


def alex(image):

    noise = tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.001, dtype=tf.float32)
    conv1 = conv((image+noise), 11, 11, 96, 4, 4, padding='VALID', name='conv1')
    norm1 = lrn(conv1, 2, 1e-04, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
    norm2 = lrn(conv2, 2, 1e-04, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

    # 4th Layer: Conv (w ReLu) splitted into two groups
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
    pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
    fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
    dropout6 = dropout(fc6, KEEP_PROB)

    # 7th Layer: FC (w ReLu) -> Dropout
    fc7 = fc(dropout6, 4096, 4096, name='fc7')
    flat = tf.reshape(fc7, [-1, VIEW_SIZE*4096])

    fc8 = fc(flat, 4096*VIEW_SIZE, 32, name='e_fc8')

    return fc8

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
             padding='SAME', groups=1):
    """Create a convolution layer.
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1],
                                             padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height,
                                                        filter_width,
                                                        input_channels / groups,
                                                        num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

    if groups == 1:
        conv = convolve(x, weights)

    # In the cases of multiple groups, split inputs & weights and
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,
                                     value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)

    return relu

def fc(x, num_in, num_out, name, relu=True, getWeights=False):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                      trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        if getWeights:
            return relu, weights
        else:
            return relu
    else:
        if getWeights:
            return act, weights
        else:
            return act

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
                 padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                  alpha=alpha, beta=beta,
                                                  bias=bias, name=name)

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

def load_data(POS_COUNT, NEG_COUNT, counter):

    label = []
    data1 = []
    data2 = []

    for i in range(STEP_SIZE):
        POS_INDEX = counter % POS_COUNT
        NEG_INDEX = counter % NEG_COUNT

        if POS_INDEX == 0:
            shuffle(POS_LIST)
        if NEG_INDEX == 0:
            shuffle(NEG_LIST)

        pos_name1 = glob.glob(DATA_DIR + POS_LIST[POS_INDEX][0] + '/' + str(POS_LIST[POS_INDEX][1]) + '_*.png')
        pos_name1.sort()
        pos_name2 = glob.glob(DATA_DIR + POS_LIST[POS_INDEX][0] + '/' + str(POS_LIST[POS_INDEX][2]) + '_*.png')
        pos_name2.sort()
        neg_name1 = glob.glob(DATA_DIR + NEG_LIST[NEG_INDEX][0] + '/' + str(NEG_LIST[NEG_INDEX][1]) + '_*.png')
        neg_name1.sort()
        neg_name2 = glob.glob(DATA_DIR + NEG_LIST[NEG_INDEX][0] + '/' + str(NEG_LIST[NEG_INDEX][2]) + '_*.png')
        neg_name2.sort()
        counter = counter + 1

        pos1_tmp = []
        pos2_tmp = []
        neg1_tmp = []
        neg2_tmp = []

        for j in range(VIEW_SIZE):
            pos_img1 = scipy.misc.imread(pos_name1[j], mode='RGB')/255.0
            pos_img2 = scipy.misc.imread(pos_name2[j], mode='RGB')/255.0
            neg_img1 = scipy.misc.imread(neg_name1[j], mode='RGB')/255.0
            neg_img2 = scipy.misc.imread(neg_name2[j], mode='RGB')/255.0
            pos1_tmp.append(pos_img1)
            pos2_tmp.append(pos_img2)
            neg1_tmp.append(neg_img1)
            neg2_tmp.append(neg_img2)

        pos1_tmp = np.array(pos1_tmp).reshape((4, 227, 227, 3))
        pos2_tmp = np.array(pos2_tmp).reshape((4, 227, 227, 3))
        neg1_tmp = np.array(neg1_tmp).reshape((4, 227, 227, 3))
        neg2_tmp = np.array(neg2_tmp).reshape((4, 227, 227, 3))

        data1.append(pos1_tmp)
        data2.append(pos2_tmp)
        data1.append(neg1_tmp)
        data2.append(neg2_tmp)
        label.append([1])
        label.append([0])

    data1 = np.array(data1).reshape((2*VIEW_SIZE*STEP_SIZE, 227, 227, 3))
    data2 = np.array(data2).reshape((2*VIEW_SIZE*STEP_SIZE, 227, 227, 3))
    label = np.array(label).reshape((2*STEP_SIZE, 1))

    return data1, data2, label

def load_data_test(filename):
    data = []

    image_list = glob.glob(DATA_DIR + filename + '/*.png')
    for i in range(len(image_list)/VIEW_SIZE):
        name = DATA_DIR + filename + '/' + str(i+1) + '_*.png'

        list_ = glob.glob(name)
        if len(list_) < VIEW_SIZE:
            continue
        list_.sort()
        for j in range(VIEW_SIZE):
            img = scipy.misc.imread(list_[j], mode='RGB')/255.0
            data.append(img)

    data = np.array(data)
    data = data.reshape((-1, 227, 227, 3))
    return data

def load(saver, sess):

    print(" [*] Reading checkpoint...")
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(CHECKPOINT_DIR, ckpt_name))
        return True
    else:
        return False

def save(saver, sess, step):

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    model_name = "net.model"
    saver.save(sess, os.path.join(CHECKPOINT_DIR, model_name), global_step=step)

def build_list():
    print 'Building Training List'
    image_list = next(os.walk(DATA_DIR))[1]
    POS_COUNT = 0
    NEG_COUNT = 0
    id_list = []
    grp_list = []
    for i in range(len(image_list)):
        img_list = glob.glob(DATA_DIR + image_list[i] + '/*.png')
        id_tmp = []
        grp_tmp = []
        count = 1
        while(count<len(img_list)/VIEW_SIZE):
            lists = glob.glob(DATA_DIR + image_list[i] + '/' + str(count) + '_*.png')
            if not len(lists) == 4:
                count = count + 1
                continue
            id_tmp.append(count)
            count = count + 1
            grp = lists[0].split('_')[-1].replace('.png', '')
            grp_tmp.append(grp)
        id_list.append(id_tmp)
        grp_list.append(grp_tmp)
        print str(i) + '/' + str(len(image_list)) + ': ' + image_list[i]

    for i in range(len(image_list)):
        id_tmp = id_list[i]
        grp_tmp = grp_list[i]
        for j in range(len(id_tmp)):
            for k in range(j + 1, len(id_tmp)):
                grp1 = grp_tmp[j]
                grp2 = grp_tmp[k]
                current_pair = [image_list[i], id_tmp[j], id_tmp[k]]
                if grp1 == grp2:
                    POS_LIST.append(current_pair)
                    POS_COUNT = POS_COUNT + 1
                else:
                    NEG_LIST.append(current_pair)
                    NEG_COUNT = NEG_COUNT + 1
        print str(i) + '/' + str(len(image_list)) + ': ' + image_list[i]

    print('positive count = %d'%POS_COUNT)
    print('negative count = %d'%NEG_COUNT)

    return POS_COUNT, NEG_COUNT


def train():

    POS_COUNT, NEG_COUNT = build_list()
    sess = tf.Session()

    data1 = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3])
    data2 = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3])
    label = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    with tf.variable_scope('net') as scope:
        feature1 = alex(data1)
        scope.reuse_variables()
        feature2 = alex(data2)

    d = tf.reduce_mean(tf.square(feature1 - feature2), 1, keep_dims=True)
    distance = tf.sqrt(d)
    dis_pos = label * tf.square(distance)
    dis_neg = (1 - label) * tf.square(tf.maximum((1 - distance), 0.0))
    contrastive_loss = tf.reduce_mean(dis_pos + dis_neg)

    dis_pos_sum = tf.summary.scalar('pos_dis', tf.reduce_mean(dis_pos))
    dis_neg_sum = tf.summary.scalar('neg_dis', tf.reduce_mean(dis_neg))
    contrastive_loss_sum = tf.summary.scalar('contrastive_loss', contrastive_loss)
    merged_summary = tf.summary.merge_all()

    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    vars = tf.trainable_variables()
    train_vars = [var for var in vars if 'e_' in var.name]

    optim_op = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(contrastive_loss, var_list=train_vars)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()

    if load(saver, sess):
        print('[*] Load Success...')
    else:
        print('[!] Load Failed!')

    start_time = time.time()

    counter = 0

    while(counter < MAX_ITER):

        data_input1, data_input2, label_input = load_data(POS_COUNT, NEG_COUNT, STEP_SIZE*counter)

        _, pos, neg, contrast, summary = sess.run([optim_op, dis_pos, dis_neg, contrastive_loss, merged_summary],
                                                 feed_dict={data1: data_input1, data2: data_input2, label: label_input})

        writer.add_summary(summary, counter)
        current_time = time.time()
        print('step: %8d, time: %.4f, pos_dis: %.8f, neg_dis: %.8f, final loss: %.8f'
                      % (counter, current_time - start_time, np.mean(pos), np.mean(neg), contrast))

        start_time = current_time

        if np.mod(counter, 1000) == 1:
            saver = tf.train.Saver()
            save(saver, sess, counter)

        counter = counter + 1

def test():

    sess = tf.Session()

    data = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3])

    with tf.variable_scope('net') as scope:
        feature = alex(data)

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()

    if load(saver, sess):
        print('[*] Load Success...')
    else:
        print('[!] Load Failed!')

    files = next(os.walk(DATA_DIR))[1]
    num_image = len(files)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i in range(num_image):

        filename = files[i]

        data_input = load_data_test(filename)

        result = sess.run(feature, feed_dict={data: data_input})

        np.savetxt(OUTPUT_DIR + filename + '.txt', result)

        print('testing image: %8d  %s' % (i, filename))


def main():

    if PHASE == 'train':
        train()
    else:
        test()

if __name__ == '__main__':
    main()
