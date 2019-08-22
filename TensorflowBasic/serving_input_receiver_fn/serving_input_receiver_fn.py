"""
准备提供输入
在训练期间，input_fn() 会提取数据，并准备好数据以供模型使用。
在提供服务期间，类似地，serving_input_receiver_fn() 接受推理请求，并为模型做好准备。该函数具有以下用途：
•	在投入使用系统将向其发出推理请求的图中添加占位符。
•	添加将数据从输入格式转换为模型所预期的特征 Tensor 所需的任何额外操作。
该函数返回一个 tf.estimator.export.ServingInputReceiver 对象，该对象会将占位符和生成的特征 Tensor 打包在一起。

典型的模式是推理请求以序列化 tf.Example 的形式到达，因此 serving_input_receiver_fn() 创建单个字符串占位符来接收它们。
serving_input_receiver_fn() 接着也负责解析 tf.Example（通过向图中添加 tf.parse_example 操作）。
在编写此类 serving_input_receiver_fn() 时，您必须将解析规范传递给 tf.parse_example，
告诉解析器可能会遇到哪些特征名称以及如何将它们映射到 Tensor。解析规范采用字典的形式，
即从特征名称映射到 tf.FixedLenFeature、tf.VarLenFeature 和 tf.SparseFeature。请注意，
此解析规范不应包含任何标签或权重列，因为这些列在服务时间将不可用（与 input_fn() 在训练时使用的解析规范相反）。

"""
import tensorflow as tf

feature_spec = {'foo': tf.FixedLenFeature(...),
                'bar': tf.VarLenFeature(...)}

def serving_input_receiver_fn():
  """An input receiver that expects a serialized tf.Example."""
  serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[default_batch_size],
                                         name='input_example_tensor')
  receiver_tensors = {'examples': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

"""
tf.estimator.export.build_parsing_serving_input_receiver_fn 效用函数提供了适用于普遍情况的输入接收器
==注意：在使用 Predict API 和本地服务器训练要投入使用的模型时，并不需要解析步骤，因为该模型将接收原始特征数据。==
即使您不需要解析或其他输入处理，也就是说，如果服务系统直接提供特征 Tensor，
您仍然必须提供一个 serving_input_receiver_fn() 来为特征 Tensor 创建占位符并在其中传递占位符。
tf.estimator.export.build_raw_serving_input_receiver_fn 效用函数实现了这一功能。
"""


"""
如果这些效用函数不能满足您的需求，您可以自由编写 serving_input_receiver_fn()。
可能需要此方法的一种情况是，如果您训练的 input_fn() 包含某些必须在服务时间重演的预处理逻辑。
为了减轻训练服务倾斜的风险，我们建议将这种处理封装在一个函数内，此函数随后将从 input_fn() 和 serving_input_receiver_fn() 两者中被调用。
请注意，serving_input_receiver_fn() 也决定了签名的输入部分。也就是说，在编写 serving_input_receiver_fn() 时，
必须告诉解析器哪些有哪些签名可能出现，以及如何将它们映射到模型的预期输入。相反，签名的输出部分由模型决定。

"""