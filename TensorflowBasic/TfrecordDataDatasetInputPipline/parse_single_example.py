"""
数据与features_col的对接
"""
import tensorflow as tf
def parse_exmp(serial_exmp):
  features = {
    "click": tf.FixedLenFeature([], tf.int64),
    "behaviorBids": tf.FixedLenFeature([20], tf.int64),
    "behaviorCids": tf.FixedLenFeature([20], tf.int64),
    "behaviorC1ids": tf.FixedLenFeature([10], tf.int64),
    "behaviorSids": tf.FixedLenFeature([20], tf.int64),
    "behaviorPids": tf.FixedLenFeature([20], tf.int64),
    "productId": tf.FixedLenFeature([], tf.int64),
    "sellerId": tf.FixedLenFeature([], tf.int64),
    "brandId": tf.FixedLenFeature([], tf.int64),
    "cate1Id": tf.FixedLenFeature([], tf.int64),
    "cateId": tf.FixedLenFeature([], tf.int64),
    "tab": tf.FixedLenFeature([], tf.string),
    "matchType": tf.FixedLenFeature([], tf.int64)
  }
  feats = tf.parse_single_example(serial_exmp, features=features)
  labels = feats.pop('click')
  return feats, labels

def train_input_fn(filenames, batch_size, shuffle_buffer_size):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(parse_exmp, num_parallel_calls=100)
  # Shuffle, repeat, and batch the examples.
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat().batch(batch_size)
  return dataset