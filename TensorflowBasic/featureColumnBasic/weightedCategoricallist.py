import tensorflow as tf
import  os
from tensorflow.python.feature_column.feature_column import _LazyBuilder
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'

"""
color和weight就是两个key，在加工tfrecord时候，不用改变数据
"""
features = {'color': [["410387","415955","412596","416526","416805","408844","418514","411611","415266"], ["410387","415955","412596","416526","416805","408844","418514","411611","415266"]],
                  'weight': [ [44.0,33.0,17.0,6.0,3.0,2.0,1.0,1.0,1.0], [44.0,33.0,17.0,6.0,3.0,2.0,1.0,1.0,1.0] ]  }

color_f_c = tf.feature_column.categorical_column_with_hash_bucket(
        key='color',
        hash_bucket_size=40,
        dtype=tf.string
    )
column = tf.feature_column.weighted_categorical_column(color_f_c, 'weight')

indicator = tf.feature_column.indicator_column(column)
tensor = tf.feature_column.input_layer(features, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))
