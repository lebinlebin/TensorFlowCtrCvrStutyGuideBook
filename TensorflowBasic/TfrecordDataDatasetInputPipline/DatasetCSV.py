"""
1. 解析CSV文件
假设我们有一个Tab分隔4个字段的文件,则可用如下的代码解析并生成dataset。
以下代码主要利用tf.decode_csv函数来把CSV文件记录转换为Tensors列表,每一列对应一个Tensor。
"""

import tensorflow as tf
import numpy as np

_CSV_COLUMNS = ['field1', 'field2', 'field3', 'field4']
_CSV_COLUMN_DEFAULTS=[[''], [''], [0.0], [0.0]]
def input_fn(data_file, shuffle, batch_size):
  def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS, field_delim='\t')
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('ctr_flag')
    return features, tf.equal(labels, '1.0')
  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)
  if shuffle: dataset = dataset.shuffle(buffer_size=100000)
  dataset = dataset.map(parse_csv, num_parallel_calls=100)
  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat()
  dataset = dataset.batch(batch_size)
  return dataset
