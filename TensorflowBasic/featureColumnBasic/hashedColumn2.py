"""
序列类特征设计完成测试
"""
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

colors = {'colors': ['green','red','red','blue','blue','yellow','pink','indigo']}
colors2 = {'colors': [['green','green'],['red','blue'],['yellow','pink']]}
column = tf.feature_column.categorical_column_with_hash_bucket(
        key='colors',
        hash_bucket_size=10,
        dtype=tf.string
    )
indicator = tf.feature_column.indicator_column(column)
tensorOnlyOne = tf.feature_column.input_layer(colors, [indicator])
tensorList = tf.feature_column.input_layer(colors2, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print("单个元素类型特征>>>>>>>>>>>>>>>")
    print(session.run([tensorOnlyOne]))
    print("List类型特征>>>>>>>>>>>>>>>")
    print(session.run([tensorList]))


"""
colors = {'colors': [['green','green'],['red','blue'],['yellow','pink']]}

[array([[0., 0., 0., 0., 2., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32)]
"""

"""
[array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)]
"""