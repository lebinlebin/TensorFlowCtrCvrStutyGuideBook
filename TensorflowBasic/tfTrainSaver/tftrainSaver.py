"""
保存和恢复

tf.train.Saver 类提供了保存和恢复模型的方法。通过 tf.saved_model.simple_save 函数可轻松地保存适合投入使用的模型。
Estimator 会自动保存和恢复 model_dir 中的变量。
保存和恢复变量
TensorFlow 变量是表示由程序操作的共享持久状态的最佳方法。
tf.train.Saver 构造函数会针对图中所有变量或指定列表的变量将 save 和 restore 操作添加到图中。
Saver 对象提供了运行这些操作的方法，并指定写入或读取检查点文件的路径。
Saver 会恢复已经在模型中定义的所有变量。如果您在不知道如何构建图的情况下加载模型（例如，如果您要编写用于加载各种模型的通用程序），
那么请阅读本文档后面的保存和恢复模型概述部分。
TensorFlow 将变量保存在二进制检查点文件中，这类文件会将变量名称映射到张量值。
"""

"""
保存变量
创建 Saver（使用 tf.train.Saver()）来管理模型中的所有变量。例如，以下代码段展示了如何调用 tf.train.Saver.save 方法以将变量保存到检查点文件中：

"""
import tensorflow as tf

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)

"""
恢复变量
tf.train.Saver 对象不仅将变量保存到检查点文件中，还将恢复变量。
请注意，当您恢复变量时，您不必事先将其初始化。例如，以下代码段展示了如何调用 tf.train.Saver.restore 方法以从检查点文件中恢复变量：
"""
tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())

"""
注意：并没有名为 /tmp/model.ckpt 的实体文件。它是为检查点创建的文件名的前缀。用户仅与前缀（而非检查点实体文件）互动。==
"""


"""
选择要保存和恢复的变量

如果您没有向 tf.train.Saver() 传递任何参数，则 Saver 会处理图中的所有变量。每个变量都保存在创建变量时所传递的名称下。
在检查点文件中明确指定变量名称的这种做法有时会非常有用。例如，您可能已经使用名为"weights"的变量训练了一个模型，
而您想要将该变量的值恢复到名为"params"的变量中。

有时候，仅保存或恢复模型使用的变量子集也会很有裨益。例如，您可能已经训练了一个五层的神经网络，现在您想要训练一个六层的新模型，
并重用该五层的现有权重。您可以使用 Saver 只恢复这前五层的权重。

您可以通过向 tf.train.Saver() 构造函数传递以下任一内容，轻松指定要保存或加载的名称和变量：
•	变量列表（将以其本身的名称保存）。
•	Python 字典，其中，键是要使用的名称，键值是要管理的变量。
继续前面所示的保存/恢复示例：
"""

tf.reset_default_graph()
# Create some variables.
v1 = tf.get_variable("v1", [3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", [5], initializer = tf.zeros_initializer)

# Add ops to save and restore only `v2` using the name "v2"
saver = tf.train.Saver({"v2": v2})

# Use the saver object normally after that.
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
  v1.initializer.run()
  saver.restore(sess, "/tmp/model.ckpt")
  print(">>>>>>>>>>>>>>>>>>>>>>>>>选择要保存和恢复的变量<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())


"""
检查某个检查点中的变量

我们可以使用 inspect_checkpoint 库快速检查某个检查点中的变量。

"""

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='', all_tensors=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]


"""
保存和恢复模型

使用 SavedModel 保存和加载模型 - 变量、图和图的元数据。SavedModel 是一种独立于语言且可恢复的神秘序列化格式，
使较高级别的系统和工具可以创建、使用和转换 TensorFlow 模型。
TensorFlow 提供了多种与 SavedModel 交互的方式，包括 tf.saved_model API、tf.estimator.Estimator 和命令行界面。

"""

"""
构建和加载 SavedModel
简单保存
创建 SavedModel 的最简单方法是使用 tf.saved_model.simple_save 函数：
"""

# simple_save(session,
#             export_dir,
#             inputs={"x": x, "y": y},
#             outputs={"z": z})





