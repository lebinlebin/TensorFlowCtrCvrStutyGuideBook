# coding: utf-8
import tensorflow as tf

"""
用tf.data.dataset构建input pipline 文档对应代码
"""

"""
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
每次获取一个元素
"""


print(">>>>>>>>>>>>>>>>>>>>>>>>case1: one-shot iterator<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
dataset = tf.data.Dataset.range(100)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    for i in range(100):
      value1 = sess.run(next_element)
      print(i,value1)
      assert i == value1

    """
    case2: initializable iterator
    initializable iterator在使用它之前需要你返回一个显式的iterator.initializer操作。
    虽然有些不便，但它允许你可以对dataset的定义进行参数化（parameterize），
    使用一或多个tf.placeholder()  tensors：它们可以当你初始化iterator时被feed进去。
    继续Dataset.range() 的示例：
    """
max_value = tf.placeholder(tf.int64, shape=[])
dataset = tf.data.Dataset.range(max_value)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    # Initialize an iterator over a dataset with 10 elements.
    sess.run(iterator.initializer, feed_dict={max_value: 10})
    for i in range(10):
        value2 = sess.run(next_element) #Tensor("IteratorGetNext:0", shape=(), dtype=int64)
        print(i, value2)
        assert i == value2

    # Initialize the same iterator over a dataset with 100 elements.
    sess.run(iterator.initializer, feed_dict={ max_value: 100})
    for i in range(100):
      value3 = sess.run(next_element)
      print(i,next_element)
      assert i == value3

    """
    case3: reinitializable iterator
    reinitializable iterator可以从多个不同的Dataset对象处初始化。
    例如，你可能有一个training input pipeline（它对输入图片做随机扰动来提高泛化能力）；
    以及一个validation input pipeline（它会在未修改过的数据上进行预测的评估）。
    这些pipeline通常使用不同的Dataset对象，但它们具有相同的结构（例如：对每个component相同的types和shapes）
    """
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
validation_dataset = tf.data.Dataset.range(50)
# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)
next_element = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# Run 20 epochs in which the training dataset is traversed, followed by the validation dataset.
with tf.Session() as sess:
    for _ in range(20):
      # Initialize an iterator over the training dataset.
      sess.run(training_init_op)
      for _ in range(100):
        sess.run(next_element)
      # Initialize an iterator over the validation dataset.
      sess.run(validation_init_op)
      for _ in range(50):
        sess.run(next_element)

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

    """
    从iterator消费数据
    Iterator.get_next()方法会返回一或多个tf.Tensor对象，对应于一个iterator的下一个element。
    每次这些tensors被评测时，它们会在底层的dataset中获得下一个element的value。
    （注意：类似于Tensorflow中其它的有状态对象，调用Iterator.get_next() 不会立即让iterator前移。
    相反的，你必须使用Tensorflow表达式所返回的tf.Tensor对象，传递该表达式的结果给tf.Session.run()，来获取下一个elements，并让iterator前移）
    如果iterator达到了dataset的结尾，执行Iterator.get_next() 操作会抛出一个tf.errors.OutOfRangeError。
    在这之后，iterator会以一个不可用的状态存在，如果你想进一步使用必须重新初始化它。
    """
dataset = tf.data.Dataset.range(5)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Typically `result` will be the output of a model, or an optimizer's
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
如果dataset的每个元素都具有一个嵌套的结构，Iterator.get_next()的返回值将会是以相同嵌套结构存在的一或多个tf.Tensor对象：
注意，对next1, next2, or next3的任意一个进行评估都会为所有components进行iterator。
一个iterator的一种常见consumer将包含在单个表达式中的所有components。
"""
dataset1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))
dataset2 = tf.data.Dataset.from_tensor_slices((tf.random_uniform([4]), tf.random_uniform([4, 100])))
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

iterator = dataset3.make_initializable_iterator()
with tf.Session() as sess:
    sess.run(iterator.initializer)
    next1, (next2, next3) = iterator.get_next()
