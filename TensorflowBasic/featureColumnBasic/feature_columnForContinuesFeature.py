# -*- coding: utf-8 -*-


import tensorflow as tf

#演示只有一个连续值特征列的操作
def test_one_column():                      
    price = tf.feature_column.numeric_column('price')          #定义一个特征列

    features = {'price': [[1.], [5.]]}                         #将样本数据定义为字典的类型
    net = tf.feature_column.input_layer(features, [price])     #将数据集与特征列一起输入，到input_layer生成张量
    
    with tf.Session() as sess:                                 #通过建立会话将其输出
        tt  = sess.run(net)
        print( tt)
print("test_one_column>>>>>>>>>>>>>>>>>>>>")
test_one_column()

#演示带占位符的特征列操作
def test_placeholder_column():
    price = tf.feature_column.numeric_column('price')          #定义一个特征列

    features = {'price':tf.placeholder(dtype=tf.float64)}      #生成一个value为占位符的字典
    net = tf.feature_column.input_layer(features, [price])     #将数据集与特征列一起输入，到input_layer生成张量

    with tf.Session() as sess:                                #通过建立会话将其输出
        tt  = sess.run(net, feed_dict={
                features['price']: [[1.], [5.]]
            })
        print( tt)

test_placeholder_column()





import numpy as np
print(np.shape([[[1., 2.]], [[5., 6.]]]))
print(np.shape([[3., 4.], [7., 8.]]))
print(np.shape([[3., 4.]]))
def test_reshaping():
    tf.reset_default_graph()
    price = tf.feature_column.numeric_column('price', shape=[1, 2])#定义一个特征列,并指定形状
    features = {'price': [[[1., 2.]], [[5., 6.]]]}  #传入一个3维的数组
    features1 = {'price': [[3., 4.], [7., 8.]]}     #传入一个2维的数组


    net = tf.feature_column.input_layer(features, price)           #生成特征列张量
    net1 = tf.feature_column.input_layer(features1, price)         #生成特征列张量
    with tf.Session() as sess:                                     #通过建立会话将其输出
        print(net.eval())
        print(net1.eval())

test_reshaping()

def test_column_order():
    tf.reset_default_graph()
    price_a = tf.feature_column.numeric_column('price_a')   #定义了3个特征列
    price_b = tf.feature_column.numeric_column('price_b')
    price_c = tf.feature_column.numeric_column('price_c')

    features = {                           #创建字典传入数据
          'price_a': [[1.]],
          'price_c': [[4.]],
          'price_b': [[3.]],
      }

    #生成输入层
    net = tf.feature_column.input_layer(features, [price_c, price_a, price_b])

    with tf.Session() as sess:             #通过建立会话将其输出
        print(net.eval())

test_column_order()

