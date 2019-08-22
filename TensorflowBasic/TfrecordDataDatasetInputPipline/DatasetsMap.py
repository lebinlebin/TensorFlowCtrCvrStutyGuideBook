"""
数据预处理接口：Datasets.map
不得不说，Datasets.map是个好东西。可以通过它，设计出你想要的任何形式的数据。
"""
"""
解析TFRecords数据
根据你的打包TFRecords方式，先进行解析操作。用map就行。
"""
import tensorflow as tf

# # Transforms a scalar string `example_proto` into a pair of a scalar string and
# # a scalar integer, representing an image and its label, respectively.
# def _parse_function(example_proto):
#   features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
#               "label": tf.FixedLenFeature((), tf.int32, default_value=0)}
#   parsed_features = tf.parse_single_example(example_proto, features)
#   return parsed_features["image"], parsed_features["label"]
#
# # Creates a dataset that reads all of the examples from two files, and extracts
# # the image and label features.
# filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
# dataset = tf.data.TFRecordDataset(filenames)
# dataset = dataset.map(_parse_function)


# """
# 图片数据decoding, resize
# 根据压缩的方式，定义decode方式，这一步往往和解析是写在一起的。然后map一下。
# """
#
# # Reads an image from a file, decodes it into a dense tensor, and resizes it
# # to a fixed shape.
# def _parse_function(filename, label):
#   image_string = tf.read_file(filename)
#   image_decoded = tf.image.decode_image(image_string)
#   image_resized = tf.image.resize_images(image_decoded, [28, 28])
#   return image_resized, label
#
# # A vector of filenames.
# filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
#
# # `labels[i]` is the label for the image in `filenames[i].
# labels = tf.constant([0, 37, ...])
#
# dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
# dataset = dataset.map(_parse_function)

"""
图片数据的normalize
一般的情况下，图片数据还是需要归一化的。
"""
def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image, label

