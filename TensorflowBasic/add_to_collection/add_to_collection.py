"""
tf.add_to_collection tf.get_collection
tf.add_to_collection–向当前计算图中添加张量集合
tf.get_collection–返回当前计算图中手动添加的张量集合
"""

#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
v1 = tf.get_variable('v1', shape=[3], initializer=tf.ones_initializer())
v2 = tf.get_variable('v2', shape=[5], initializer=tf.random_uniform_initializer(maxval=-1., minval=1., seed=0))

# 向当前计算图中添加张量集合
tf.add_to_collection('v', v1)
tf.add_to_collection('v', v2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 返回当前计算图中手动添加的张量集合
    v = tf.get_collection('v')
    print (v)
    print (v[0].eval())
    print (v[1].eval())
