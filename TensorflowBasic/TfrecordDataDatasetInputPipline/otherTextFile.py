"""
2. 解析特殊格式的文本文件
有时候我们的训练数据可能有特殊的格式,比如CVS文件其中某些字段是JSON格式的字符串,
我们要把JSON字符串的内容也解析出来,这个时候tf.decode_csv函数就不够用了。
是时候请万能函数tf.py_func上场了,tf.py_func函数能够把一个任意的python函数封装成tensorflow的op,提供了极大的灵活性,其定义如下：
tf.py_func(
    func,
    inp,
    Tout,
    stateful=True,
    name=None
)
tf.py_func的核心是一个func函数(由用户自己定义),该函数被封装成graph中的一个节点（op)。
第二个参数inp是一个由Tensor组成的list,在执行时,inp的各个Tensor的值被取出来传给func作为参数。
func的返回值会被tf.py_func转换为Tensors,这些Tensors的类型由Tout指定。当func只有一个返回值时,
Tout是一个单独的tensorflow数据类型；当func函数有多个返回值时,Tout是一个tensorflow数据类型组成的元组或列表。
参数stateful表示func函数是否有状态（产生副作用）。
在使用过程中,有几个需要注意的地方：
•	func函数的返回值类型一定要和Tout指定的tensor类型一致。
•	tf.py_func中的func是脱离Graph的,在func中不能定义可训练的参数参与网络训练(反传)。
•	tf.py_func操作只能在CPU上运行；如果使用分布式TensorFlow, tf.py_func操作必须放在与客户端相同进程的CPU设备上。
•	tf.py_func操作返回的tensors是没有定义形状（shape）的,必须调用set_shape方法为各个返回值设置shape,才能参与后续的计算。
先来看一个简单的示例,func函数接受单个参数并产生单个返回值的情况。
"""
import json
import numpy as np
import tensorflow as tf


def filter_func(line):
  fields = line.decode().split("\t")
  if len(fields) < 8:
    return False
  for field in fields:
    if not field:
      return False
  return True
"""
dataset = dataset.filter(lambda x: tf.py_func(filter_func, [x], tf.bool, False))
再来看一个稍微复杂一点的例子,该例子解析一个带有json格式字段的CSV文件,json字段被平铺开来和其他字段并列作为返回值。

"""
def parse_line(line):
  _COLUMNS = ["sellerId", "brandId", "cateId"]
  _INT_COLUMNS = ["click", "productId", "matchType", "position", "hour"]
  _FLOAT_COLUMNS = ["matchScore", "popScore", "brandPrefer", "catePrefer"]
  _STRING_COLUMNS = ["phoneResolution", "phoneBrand", "phoneOs"]
  _SEQ_COLUMNS = ["behaviorC1ids", "behaviorBids", "behaviorCids", "behaviorPids"]

  def get_content(record):
    import datetime
    fields = record.decode().split("\t")
    if len(fields) < 8:
      raise ValueError("invalid record %s" % record)
    for field in fields:
      if not field:
        raise ValueError("invalid record %s" % record)
    fea = json.loads(fields[1])
    if fea["time"]:
      dt = datetime.datetime.fromtimestamp(fea["time"])
      fea["hour"] = dt.hour
    else:
      fea["hour"] = 0
    seq_len = 10
    for x in _SEQ_COLUMNS:
      sequence = fea.setdefault(x, [])
      n = len(sequence)
      if n < seq_len:
        sequence.extend([-1] * (seq_len - n))
      elif n > seq_len:
        fea[x] = sequence[:seq_len]
      seq_len = 20

    elems = [np.int64(fields[2]), np.int64(fields[3]), np.int64(fields[4]), np.int64(fields[6]), fields[7]]
    elems += [np.int64(fea.get(x, 0)) for x in _INT_COLUMNS]
    elems += [np.float32(fea.get(x, 0.0)) for x in _FLOAT_COLUMNS]
    elems += [fea.get(x, "") for x in _STRING_COLUMNS]
    elems += [np.int64(fea[x]) for x in _SEQ_COLUMNS]
    return elems

  out_type = [tf.int64] * 4 + [tf.string] + [tf.int64] * len(_INT_COLUMNS) + [tf.float32] * len(_FLOAT_COLUMNS) + [
    tf.string] * len(_STRING_COLUMNS) + [tf.int64] * len(_SEQ_COLUMNS)
  result = tf.py_func(get_content, [line], out_type)
  n = len(result) - len(_SEQ_COLUMNS)
  for i in range(n):
    result[i].set_shape([])
  result[n].set_shape([10])
  for i in range(n + 1, len(result)):
    result[i].set_shape([20])
  columns = _COLUMNS + _INT_COLUMNS + _FLOAT_COLUMNS + _STRING_COLUMNS + _SEQ_COLUMNS
  features = dict(zip(columns, result))
  labels = features.pop('click')
  return features, labels

def my_input_fn(filenames, batch_size, shuffle_buffer_size):
  dataset = tf.data.TextLineDataset(filenames)
  dataset = dataset.filter(lambda x: tf.py_func(filter_func, [x], tf.bool, False))
  dataset = dataset.map(parse_line, num_parallel_calls=100)
  # Shuffle, repeat, and batch the examples.
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat().batch(batch_size)
  return dataset
