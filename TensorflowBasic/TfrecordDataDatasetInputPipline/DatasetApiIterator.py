import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 设置当前使用的GPU设备仅为0号设备  设备名称为'/gpu:0'
"""
用tf.data.dataset构建input pipline 文档对应代码
"""
"""
在TensorFlow 1.3版本之前,读取数据一般有两种方法：
•	使用placeholder + feed_dict读内存中的数据
•	使用文件名队列（string_input_producer）与内存队列（reader）读硬盘中的数据

Dataset API同时支持从内存和硬盘的数据读取,相比之前的两种方法在语法上更加简洁易懂。
Dataset API可以更方便地与其他高阶API配合,快速搭建网络模型。
此外,如果想要用到TensorFlow新出的Eager模式,就必须要使用Dataset API来读取数据。

Dataset可以看作是相同类型“元素”的有序列表。在实际使用时,单个“元素”可以是向量,也可以是字符串、图片,甚至是tuple或者dict。

TensorFlow的tf.random_uniform()函数的用法
tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))返回6*6的矩阵，产生于low和high之间，产生的值是均匀分布的。
"""
# import tensorflow as tf
# with tf.Session() as sess:
#     print(sess.run(tf.random_uniform((6,6), minval=-0.5,maxval=0.5, dtype=tf.float32)))


"""
tf.data.Dataset.from_tensor_slices()
我们在转化数据集时经常会使用这个函数，他的所用是切分传入的 Tensor 的第一个维度，生成相应的 dataset。
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5, 2))) 
传入的数值是一个矩阵，它的形状为(5, 2)，tf.data.Dataset.from_tensor_slices就会切分它形状上的第一个维度，
最后生成的dataset中一个含有5个元素，每个元素的形状是(2, )，即每个元素是矩阵的一行。

对于更复杂的情形，比如元素是一个python中的元组或者字典：
在图像识别中一个元素可以是｛”image”:image_tensor,”label”:label_tensor｝的形式。 
dataset = tf.data.Dataset.from_tensor_slices ( { “a”:np.array([1.0,2.0,3.0,4.0,5.0]), “b”:np.random.uniform(size=(5,2) ) } ) 
这时，函数会分别切分”a”中的数值以及”b”中的数值，最后总dataset中的一个元素就是类似于{ “a”:1.0, “b”:[0.9,0.1] }的形式。
"""


dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10])) #每行10个元素
print(dataset1.output_types)  # ==> "tf.float32"
print(dataset1.output_shapes)  # ==> "(10,)"

dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))
print(dataset2.output_types)  # ==> "(tf.float32, tf.int32)"
print(dataset2.output_shapes)  # ==> "((), (100,))"

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))
print(dataset3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(dataset3.output_shapes)  # ==> "(10, ((), (100,)))"

"""
为一个元素(element)的每个component给定names很方便，例如，如果它们表示一个训练样本的不同features。
除了tuples，你可以使用collections.namedtuple，或者一个将strings映射为关于tensors的字典来表示一个Dataset的单个元素。
"""
dataset = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
print(dataset.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(dataset.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

"""
Dataset的转换（transformations）支持任何结构的datasets。
当使用Dataset.map()，Dataset.flat_map()，以及Dataset.filter()转换时，它们会对每个element应用一个function，元素结构决定了函数的参数：
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)
"""



"""
#############################################################################################################
创建一个iterator
一旦你已经构建了一个Dataset来表示你的输入数据，下一步是创建一个Iterator来访问dataset的elements。
Dataset API当前支持四种iterator，复杂度依次递增：
•	one-shot
•	initializable
•	reinitializable
•	feedable
"""
"""
case1: one-shot iterator
one-shot iterator是最简单的iterator，它只支持在一个dataset上迭代一次的操作，
不需要显式初始化。One-shot iterators可以处理几乎所有的己存在的基于队列的input pipeline支持的情况，
但它们不支持参数化（parameterization）。使用Dataset.range()示例如下：
"""
# print(">>>>>>>>>>>>>>>>>>>>>>>>case1: one-shot iterator<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
# dataset = tf.data.Dataset.range(100)
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
#
# with tf.Session() as sess:
#     for i in range(100):
#       value = sess.run(next_element)
#       assert i == value
#       print(i,value)


"""
case2: initializable iterator
initializable iterator在使用它之前需要你返回一个显式的iterator.initializer操作。
虽然有些不便，但它允许你可以对dataset的定义进行参数化（parameterize），
使用一或多个tf.placeholder()
tensors：它们可以当你初始化iterator时被feed进去。继续Dataset.range() 的示例：
"""
# max_value = tf.placeholder(tf.int64, shape=[])
# dataset = tf.data.Dataset.range(max_value)
# iterator = dataset.make_initializable_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     print("Initialize an iterator over a dataset with 10 elements.")
#     # Initialize an iterator over a dataset with 10 elements.
#     sess.run(iterator.initializer, feed_dict={max_value: 10})
#     for i in range(10):
#       value = sess.run(next_element)
#       assert i == value
#       print(i,value)
#
# print("Initialize the same iterator over a dataset with 100 elements.")
# with tf.Session() as sess:
#     # Initialize the same iterator over a dataset with 100 elements.
#     sess.run(iterator.initializer, feed_dict={max_value: 100})
#     for i in range(100):
#       value = sess.run(next_element)
#       assert i == value
#       print(i,value)


"""
case3: reinitializable iterator
reinitializable iterator可以从多个不同的Dataset对象处初始化。
例如，你可能有一个training input pipeline（它对输入图片做随机扰动来提高泛化能力）；
以及一个validation input pipeline（它会在未修改过的数据上进行预测的评估）。
这些pipeline通常使用不同的Dataset对象，但它们具有相同的结构（例如：对每个component相同的types和shapes）
"""
# # Define training and validation datasets with the same structure.
# training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
# validation_dataset = tf.data.Dataset.range(50)
#
# # A reinitializable iterator is defined by its structure. We could use the
# # `output_types` and `output_shapes` properties of either `training_dataset`
# # or `validation_dataset` here, because they are compatible.
# iterator = tf.data.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)
# next_element = iterator.get_next()
#
# training_init_op = iterator.make_initializer(training_dataset)
# validation_init_op = iterator.make_initializer(validation_dataset)
#
# with tf.Session() as sess:
#     # Run 20 epochs in which the training dataset is traversed, followed by the
#     # validation dataset.
#     for _ in range(2):
#       print("Initialize an iterator over the training dataset.")
#       # Initialize an iterator over the training dataset.
#       sess.run(training_init_op)
#       for _ in range(100):
#           print(sess.run(next_element))
#
#       # Initialize an iterator over the validation dataset.
#       print("Initialize an iterator over the validation dataset.")
#       sess.run(validation_init_op)
#       for _ in range(50):
#         print(sess.run(next_element))

"""
case4： feedable iterator
feedable iterator可以与tf.placeholder一起使用，通过熟悉的feed_dict机制，来选择在每次调用tf.Session.run所使用的Iterator，。
它提供了与reinitializable iterator相同的功能，但当你在iterators间相互切换时，它不需要你去初始化iterator。
例如：使用上述相同的training和validation样本，你可以使用tf.data.Iterator.from_string_handle来定义一个feedable iterator，
并允许你在两个datasets间切换：
"""
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(
    lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
validation_dataset = tf.data.Dataset.range(50)

# A feedable iterator is defined by a handle placeholder and its structure. We
# could use the `output_types` and `output_shapes` properties of either
# `training_dataset` or `validation_dataset` here, because they have
# identical structure.
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, training_dataset.output_types, training_dataset.output_shapes)
next_element = iterator.get_next()

# You can use feedable iterators with a variety of different kinds of iterator
# (such as one-shot and initializable iterators).
training_iterator = training_dataset.make_one_shot_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()


with tf.Session() as sess:
    # The `Iterator.string_handle()` method returns a tensor that can be evaluated
    # and used to feed the `handle` placeholder.
    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    # Loop forever, alternating between training and validation.
    while True:
      # Run 200 steps using the training dataset. Note that the training dataset is
      # infinite, and we resume from where we left off in the previous `while` loop
      # iteration.
      for _ in range(200):
        sess.run(next_element, feed_dict={handle: training_handle})

      # Run one pass over the validation dataset.
      sess.run(validation_iterator.initializer)
      for _ in range(50):
        sess.run(next_element, feed_dict={handle: validation_handle})



