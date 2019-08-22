"""
本文件用于测试tfrecord转化为原始数据
"""
import tensorflow as tf
import matplotlib.pyplot as plt

tfrecord_path_none = "E:\CODEING\codeingForSelfStudy\CTR_CVR_DEEPLEARNING\ESMM_Online\data\SAMPLETFrecord2"
# 返回一个QueueRunner，里面有一个FIFOQueue，包含不同的tfrecord文件名队列
filename_queues = tf.train.string_input_producer([tfrecord_path_none])
# 定义不同压缩选项的TFRecordReader
reader_none = tf.TFRecordReader(options=None)
# 读取不同的tfrecord文件
_,serialized_example_none = reader_none.read(filename_queues)
# 根据key名字得到保存的features字典
features_none = tf.parse_single_example(serialized_example_none,
features={
"key":tf.FixedLenFeature([], tf.float32),
"label":tf.FixedLenFeature([], tf.int64),
"height":tf.FixedLenFeature([], tf.int64),
"image_raw":tf.FixedLenFeature([], tf.string)
})


# 保存时是以image原始格式数据，读出来后，还需要解码，只显示features_gzip的图片即可
image = tf.image.decode_jpeg(features_gzip['image_raw'], channels=3)
# 启用队列协调管理器，并使用tf.train.start_queue_runners启动队列文件线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# 获取不同tfrecord文件保存的float_val的值，正确的值应该是9.99/8.88/6.66
sess = tf.Session()
image,float_val_none,float_val_zlib,float_val_gzip = sess.run([image, features_none['float_val'], features_zlib['float_val'], features_gzip['float_val']])
# 关闭线程以及会话
coord.request_stop()
coord.join(threads)
sess.close()
# 打印获取到不同tfrecord文件里的float_val，以验证正确性
print(float_val_none)
print(float_val_zlib)
print(float_val_gzip)
# 显示读取到的图片
plt.imshow(image)
plt.title("beautiful view")
plt.show()

print("finish to read data from tfrecord file!")
