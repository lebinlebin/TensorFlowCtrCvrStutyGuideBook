"""
Tensorflow 低阶API 简介
________________________________________
简介
本指南旨在指导您使用低级别 TensorFlow API (TensorFlow Core) 开始编程。您可以学习执行以下操作
•	管理您自己的 TensorFlow 程序 (tf.Graph) 和 TensorFlow 运行时 (tf.Session)，而不是依靠 Estimator 来管理它们。
•	使用 tf.Session 运行 TensorFlow 操作。
•	在此低级别环境中使用高级别组件（数据集、层和 feature_columns）。
•	构建自己的训练循环，而不是使用 Estimator 提供的训练循环。

我们建议尽可能使用更高阶的 API 构建模型。以下是 TensorFlow Core 为何很重要的原因：
•	如果您能够直接使用低阶 TensorFlow 操作，实验和调试都会更直接。
•	在使用更高阶的 API 时，能够理解其内部工作原理。

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
"""
张量值
TensorFlow 中的核心数据单位是张量。一个张量由一组形成阵列（任意维数）的原始值组成。
张量的阶是它的维数，而它的形状是一个整数元组，指定了阵列每个维度的长度。
以下是张量值的一些示例：
3. # a rank 0 tensor; a scalar with shape [],
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]


TensorFlow 使用 numpy 阵列来表示张量值。


"""


a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0) # also tf.float32 implicitly
total = a + b
print(a)
print(b)
print(total)

sess = tf.Session()
print(sess.run(total))
print(sess.run({'ab':(a, b), 'total':total}))


vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
"""
另请注意，feed_dict 参数可用于覆盖图中的任何张量。占位符和其他 tf.Tensors 的唯一不同之处在于如果没有为占位符提供值，那么占位符会抛出错误。
"""


my_data = [
    [0, 1,],
    [2, 3,],
    [4, 5,],
    [6, 7,],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()

while True:
  try:
    print(sess.run(next_item))
  except tf.errors.OutOfRangeError:
    break

"""
如果 Dataset 依赖于有状态操作，则可能需要在使用迭代器之前先初始化它，如下所示：
"""
r = tf.random_normal([10,3])
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
  try:
    print(sess.run(next_row))
  except tf.errors.OutOfRangeError:
    break


"""
层
可训练的模型必须修改图中的值，以便在输入相同值的情况下获得新的输出值。将可训练参数添加到图中的首选方法是层。
层将变量和作用于它们的操作打包在一起。例如，密集连接层会对每个输出对应的所有输入执行加权和，并应用激活函数（可选）。连接权重和偏差由层对象管理。
创建层
下面的代码会创建一个 Dense 层，该层会接受一批输入矢量，并为每个矢量生成一个输出值。要将层应用于输入值，请将该层当做函数来调用。例如：
"""
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)

init = tf.global_variables_initializer()
sess.run(init)


"""
特征列

使用特征列进行实验的最简单方法是使用 tf.feature_column.input_layer 函数。
此函数只接受密集列作为输入，因此要查看类别列的结果，您必须将其封装在
 tf.feature_column.indicator_column 中。例如：
"""
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>测试特征列显示原始数据<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
features = {
    'sales' : [[5], [10], [8], [9]],
    'department': ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'department', ['sports', 'gardening'])
department_column = tf.feature_column.indicator_column(department_column)

columns = [
    tf.feature_column.numeric_column('sales'),
    department_column
]

inputs = tf.feature_column.input_layer(features, columns)
var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()
sess = tf.Session()
sess.run((var_init, table_init))
print(sess.run(inputs))




"""
手动训练一个小型回归模型吧。
"""

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for i in range(100):
  _, loss_value = sess.run((train, loss))
  print(loss_value)

print(sess.run(y_pred))
