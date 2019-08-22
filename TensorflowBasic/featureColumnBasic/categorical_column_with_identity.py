"""
tensorflow-gpu  bug
这里对于windows版本的tensorflow-gpu版本对于
tf.feature_column.indicator_column
有bug。
同时tf.one_hot()也有这个bug
"""

"""
分类识别列Categorical identity column
很多数据都不是数字格式的，比如动物的类别“猫狗牛羊”、商品的类别“食品服装数码”、人的姓氏“张王李赵”...这些都是文字格式的。
但是，Tensorflow只能处理数字。
我们必须把字符名称变为数字模式，或者说我们必须用数字来表示文字。
参照上面的分箱的方法，我们可以创建很多箱子表示各种动物，把每个种类动物名称写在卡片上，放到对应的箱子里。
假设我们有4种宠物分类：猫，狗，兔子，猪，对应列表[a1,a2,a3,a4]那么就有:
宠物类别的独热编码
语法格式
categorical_column_with_identity(
    key,
    num_buckets,
    default_value=None
)
测试代码
import tensorflow as tf
pets = {'pets': [2,3,0,1]}  #猫0，狗1，兔子2，猪3  categorical_column_with_identity需要对原始数据进行一个数字化表示
column = tf.feature_column.categorical_column_with_identity(
    key='pets',
    num_buckets=4)
indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(pets, [indicator])
with tf.Session() as session:
        print(session.run([tensor]))
运行输出结果
[array([[0., 0., 1., 0.], #兔子
       [0., 0., 0., 1.], #猪
       [1., 0., 0., 0.], #猫
       [0., 1., 0., 0.]], dtype=float32)] #狗



 Use this when your inputs are integers in the range `[0, num_buckets)`, and
  you want to use the input value itself as the categorical ID. Values outside
  this range will result in `default_value` if specified, otherwise it will
  fail.


"""

import tensorflow as tf
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #不用GPU运行即可运行成功

pets = {'pets': [2,3,0,1]}  #猫0，狗1，兔子2，猪3  categorical_column_with_identity需要对原始数据进行一个数字化表示
column = tf.feature_column.categorical_column_with_identity(
    key='pets',
    num_buckets=4)
with tf.device('/cpu:0'):
    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])
with tf.Session() as session:
        print(session.run([tensor]))

