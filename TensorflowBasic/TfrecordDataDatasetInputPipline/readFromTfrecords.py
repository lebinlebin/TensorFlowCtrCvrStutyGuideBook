"""
从tfrecords读取数据
Dataset API支持多种文件格式，因此你可以处理超过内存大小的大数据集。
例如，TFRecord文件格式是一种简单的面向记录的二进制格式，许多TensorFlow应用都用它来做训练数据。
tf.data.TFRecordDataset类允许你在一或多个TFRecord文件的内容上进行流化，将它们作为input pipeline的一部分：

"""
import tensorflow as tf
# Creates a dataset that reads all of the examples from two files.
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
dataset = tf.data.TFRecordDataset(filenames)
