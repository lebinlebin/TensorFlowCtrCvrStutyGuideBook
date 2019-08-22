import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

colors2 = {'colors': ['green','red','red','blue','blue','yellow','pink','indigo']}
colors = {'colors': [['green','green'],['red','blue'],['yellow','pink','indigo']]}
column = tf.feature_column.categorical_column_with_hash_bucket(
        key='colors',
        hash_bucket_size=10,
        dtype=tf.string
    )
"""
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],  # 4行样本 shape=[4,2]
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本  shape=[4,1]

"""
indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(colors, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))


"""
[array([[0., 0., 0., 0., 1., 0., 0., 0., 0., 0.], #green
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], #red
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], #red
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], #blue
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], #blue
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], #yellow
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], #pink
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], #indigo




[array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)]
"""