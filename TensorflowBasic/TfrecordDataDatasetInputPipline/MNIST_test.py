"""
6. 举个栗子
这里我们用mnist作为例子。
6.1 写入tfrecords文件
写入tfrecords文件的方法很多，可能会出现很多不可思议的类型错误。
"""
# !/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import logging
import os
import numpy as np

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
logging.basicConfig(level=logging.INFO)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_tfrecords():
    # 注意读取的类型，dtype
    mnist = input_data.read_data_sets("./data", dtype=tf.uint8, reshape=True)
    tf.logging.set_verbosity(old_v)

    # convert train data to tfrecords
    data_train = mnist.train
    images = data_train.images  # list
    labels = data_train.labels
    num_examples = data_train.num_examples

    assert images.shape[0] == num_examples
    num_examples, feature_length = images.shape

    # tfrecords output path
    # os.rmdir('./data/')
    # os.mkdir('./data')
    outpath = './data/mnist_train.tfrecords'

    print ("begin writing data into tfrecords.....")
    writer = tf.python_io.TFRecordWriter(outpath)
    for indx in range(num_examples):
        image_raw = images[indx].tostring()
        logging.info('images-{} write in '.format(indx))

        # build a example proto for an image example
        example = tf.train.Example(features=tf.train.Features( feature={
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(int(labels[indx]))
        }))

        # write in tfrecords
        writer.write(example.SerializeToString())
    print ("writing over !")
    writer.close()
    return

convert_tfrecords()


"""
6.2 map函数数据预处理
"""
def _parse_data(example_proto):
    features = { "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
               "label": tf.FixedLenFeature((),tf.int64,default_value=0)}
    parsed_features = tf.parse_single_example(example_proto, features)

    img = parsed_features["image_raw"]
    img = tf.decode_raw(img, tf.uint8)
    img = tf.cast(img,tf.float32) * (1./255.) - 0.5
    label = parsed_features["label"]
    return img,label

"""
6.3 处理多个epochs
# just repeat 一下就ok
dataset = dataset.repeat(10)
"""

"""
6.4 random shuffling一下
Dataset.shuffle() 转换会与tf.RandomShuffleQueue使用相同的算法对输入数据集进行随机shuffle：
它会维持一个固定大小的buffer，并从该buffer中随机均匀地选择下一个元素：
dataset = dataset.shuffle(buffer_size=10000)
"""

"""
6.5 整理输出来看看
我这里就没shuffling了。凑合看吧~
"""
filenames = ['./data/mnist_train.tfrecords']
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_data)
dataset = dataset.repeat(10)
dataset = dataset.batch(32)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

sess = tf.Session()
for i in range(10):
    img,label = sess.run(next_element)
    print (img.shape,label)
sess.close()
