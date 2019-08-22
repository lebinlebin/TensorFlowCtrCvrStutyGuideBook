import tensorflow as tf
"""
简单的batching
batching的最简单方式是，将数据集上n个连续的elements进行stack成单个elements。
Dataset.batch() 转换可以精准地做到这一点，它使用与tf.stack() 操作相同的constraints，
应用在元素的每个component上：例如，对于每个元素i，所有元素必须具有一个相同shape的tensor：
"""
# inc_dataset = tf.data.Dataset.range(100)
# dec_dataset = tf.data.Dataset.range(0, -100, -1)
# dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
# batched_dataset = dataset.batch(4)
#
# iterator = batched_dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
# with tf.Session() as sess:
#     print(sess.run(next_element))  # ==> ([0, 1, 2,   3],   [ 0, -1,  -2,  -3])
#     print(sess.run(next_element))  # ==> ([4, 5, 6,   7],   [-4, -5,  -6,  -7])
#     print(sess.run(next_element))  # ==> ([8, 9, 10, 11],   [-8, -9, -10, -11])

"""
使用padding打包tensors
上面的方法需要相同的size。
然而，许多模型（比如：序列模型）的输入数据的size多种多样（例如：序列具有不同的长度）
为了处理这种情况，Dataset.padded_batch() 转换允许你将不同shape的tensors进行batch，
通过指定一或多个dimensions，在其上进行pad。

Dataset.padded_batch() 转换允许你为每个component的每个dimension设置不同的padding，
它可以是可变的长度（在样本上指定None即可）或恒定长度。你可以对padding值（缺省为0.0）进行override。
  
  # Output tensor has shape [2, 3].
  fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
                       
  x = tf.constant([1.8, 2.2], dtype=tf.float32)
  tf.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
"""

dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    print(sess.run(next_element))  # ==> [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 3, 3]]
    print(sess.run(next_element))  # ==> [[4, 4, 4, 4, 0, 0, 0],
                                   #      [5, 5, 5, 5, 5, 0, 0],
                                   #      [6, 6, 6, 6, 6, 6, 0],
                                   #      [7, 7, 7, 7, 7, 7, 7]]
