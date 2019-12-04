import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
"""
2.3 从iterator消费数据
Iterator.get_next()方法会返回一或多个tf.Tensor对象，对应于一个iterator的下一个element。
每次这些tensors被评测时，它们会在底层的dataset中获得下一个element的value。
（注意：类似于Tensorflow中其它的有状态对象，调用Iterator.get_next() 不会立即让iterator前移。
相反的，你必须使用Tensorflow表达式所返回的tf.Tensor对象，传递该表达式的结果给tf.Session.run()，
来获取下一个elements，并让iterator前移）
如果iterator达到了dataset的结尾，执行Iterator.get_next() 操作会抛出一个tf.errors.OutOfRangeError。
在这之后，iterator会以一个不可用的状态存在，如果你想进一步使用必须重新初始化它。
"""
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer’s
# training operation.
result = tf.add(next_element, next_element)
with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run(result))  # ==> "0"
    print(sess.run(result))  # ==> "2"
    print(sess.run(result))  # ==> "4"
    print(sess.run(result))  # ==> "6"
    print(sess.run(result))  # ==> "8"
    try:
      sess.run(result)
    except tf.errors.OutOfRangeError:
      print("End of dataset")  # ==> "End of dataset"


"""
如果dataset的每个元素都具有一个嵌套的结构，Iterator.get_next()的返回值
将会是以相同嵌套结构存在的一或多个tf.Tensor对象：
"""
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    next1, (next2, next3) = iterator.get_next()
    print((sess.run(next1),(sess.run(next2),sess.run(next3))))
"""
注意，对next1, next2, or next3的任意一个进行评估都会为所有components进行iterator。
一个iterator的一种常见consumer将包含在单个表达式中的所有components。
"""