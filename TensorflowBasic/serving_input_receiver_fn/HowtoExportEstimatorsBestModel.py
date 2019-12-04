feature_spec = {
        'shape': tf.VarLenFeature(tf.int64),
        'image_raw': tf.FixedLenFeature((), tf.string),
        'label_raw': tf.FixedLenFeature((43), tf.int64)
    }

def serving_input_receiver_fn():
  serialized_tf_example = tf.placeholder(
dtype=tf.string,
            shape=[120, 120, 3],#120x120picture and 3 channel
            name='input_example_tensor')
  receiver_tensors = {'image': serialized_tf_example}
  features = tf.parse_example(serialized_tf_example, feature_spec)
  return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


def serving_input_receiver_fn():
    feature_spec = {
            'image': tf.FixedLenFeature((), tf.string)
        }
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
