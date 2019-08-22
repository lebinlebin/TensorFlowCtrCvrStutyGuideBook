"""
指定自定义模型的输出
编写自定义 model_fn 时，必须填充 export_outputs 元素（属于 tf.estimator.EstimatorSpec 返回值）。
这是 {name: output} 描述在服务期间输出和使用的输出签名的词典。
在进行单一预测的通常情况下，该词典包含一个元素，而且 name 不重要。
在一个多头模型中，每个头部都由这个词典中的一个条目表示。在这种情况下，name 是一个您所选择的字符串，用于在服务时间内请求特定头部。
每个 output 值必须是一个 ExportOutput 对象，例如 tf.estimator.export.ClassificationOutput、tf.estimator.export.RegressionOutput
或 tf.estimator.export.PredictOutput。
这些输出类型直接映射到 TensorFlow Serving API，并确定将支持哪些请求类型。
==注意：在多头情况下，系统将为从 model_fn 返回的 export_outputs 字典的每个元素生成 SignatureDef，
这些元素都以相同的键命名。这些 SignatureDef 仅在输出（由相应的 ExportOutput 条目提供）方面有所不同。
输入始终是由 serving_input_receiver_fn 提供的。推理请求可以按名称指定头部。
一个头部必须使用 signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY 命名，
表示在推理请求没有指定 SignatureDef 时将提供哪一个 SignatureDef。==


"""

"""
导出 Estimator
要导出已训练的 Estimator，请调用 tf.estimator.Estimator.export_savedmodel 并提供导出基本路径和 serving_input_receiver_fn。
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn,
                            strip_default_attrs=True)
该方法通过以下方式构建新图：首先调用 serving_input_receiver_fn() 以获得特征 Tensor，然后调用此 Estimator 的 model_fn()，
以基于这些特征生成模型图。它会重新启动 Session，并且默认情况下会将最近的检查点恢复到它（如果需要，可以传递不同的检查点）。
最后，它在给定的 export_dir_base（即 export_dir_base/<timestamp>）下面创建一个带时间戳的导出目录，
并将 SavedModel 写入其中，其中包含从此会话中保存的单个 MetaGraphDef。
==注意：您负责对先前的导出操作进行垃圾回收。否则，连续导出将在 export_dir_base 下累积垃圾资源。==
"""