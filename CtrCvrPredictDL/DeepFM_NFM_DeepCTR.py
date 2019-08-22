# coding: utf-8
"""
# reference：
# * [《广告点击率预估是怎么回事？》](https://zhuanlan.zhihu.com/p/23499698)
# * [从ctr预估问题看看f(x)设计—DNN篇](https://zhuanlan.zhihu.com/p/28202287)
# 同样以criteo数据为例，我们来看看深度学习的应用。
"""

"""
特征工程
特征工程是比较重要的数据处理过程，这里对criteo数据依照[paddlepaddle做ctr预估特征工程]
(https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py)完成特征工程
特征工程参考(https://github.com/PaddlePaddle/models/blob/develop/deep_fm/preprocess.py)完成
-对数值型特征，normalize处理
-对类别型特征，对长尾(出现频次低于200)的进行过滤
"""
import os
import sys
import random
import collections
import argparse
from multiprocessing import Pool as ThreadPool

# 13个连续型列，26个类别型列
continous_features = range(1, 14)
categorial_features = range(14, 40)

# 对连续值进行截断处理(取每个连续值列的95%分位数)  13个连续值得数 分别统计13个特征得大于95%分位数得值
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


class CategoryDictGenerator:
    """
    类别型特征编码字典
    对类别得频次较低得去掉
    """
    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    # cutoff截断
    def build(self, datafile, categorial_features, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0
    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res
    def dicts_sizes(self):
        return map(len, self.dicts)


class ContinuousFeatureGenerator:
    """
    对连续值特征做最大最小值normalization
    """
    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[continous_features[i]]
                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


def preprocess(input_dir, output_dir):
    """
    对连续型和类别型特征进行处理
    """
    dists = ContinuousFeatureGenerator(len(continous_features))
    dists.build(input_dir + 'train.txt', continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(input_dir + 'train.txt', categorial_features, cutoff=150)

    output = open(output_dir + 'feature_map','w')
    for i in continous_features:
        output.write("{0} {1}\n".format('I'+str(i), i))
    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [dists.num_feature]
    for i in range(1, len(categorial_features)+1):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)
        for key, val in dicts.dicts[i-1].iteritems():
            output.write("{0} {1}\n".format('C'+str(i)+'|'+key, categorial_feature_offset[i - 1]+val+1))
    random.seed(0)

    # 90%的数据用于训练，10%的数据用于验证
    with open(output_dir + 'tr.libsvm', 'w') as out_train:
        with open(output_dir + 'va.libsvm', 'w') as out_valid:
            with open(input_dir + 'train.txt', 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_vals = []
                    for i in range(0, len(continous_features)):
                        val = dists.gen(i, features[continous_features[i]])
                        feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        feat_vals.append(str(val) + ':1')

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    with open(output_dir + 'te.libsvm', 'w') as out:
        with open(input_dir + 'test.txt', 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    feat_vals.append(str(continous_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1]) + categorial_feature_offset[i]
                    feat_vals.append(str(val) + ':1')

                out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))



input_dir = './data/criteo_data/'
output_dir = './data/criteo_data/'
print("开始数据处理与特征工程...")
preprocess(input_dir, output_dir)


"""
DeepFM
reference：[常见的计算广告点击率预估算法总结](https://zhuanlan.zhihu.com/p/29053940)
DeepFM结合了wide and deep和FM，其实就是把PNN和WDL结合了。原始的Wide and Deep，Wide的部分只是LR，构造线性关系，
Deep部分建模更高阶的关系，所以在Wide and Deep中还需要做一些特征的东西，如Cross Column的工作，
FM是可以建模二阶关系达到Cross column的效果，DeepFM就是把FM和NN结合，无需再对特征做诸如Cross Column的工作了。

DeepFM相对于FNN、PNN，能够利用其Deep部分建模更高阶信息（二阶以上），而相对于Wide and Deep能够减少特征工程的部分工作，
wide部分类似FM建模一、二阶特征间关系， 算是NN和FM的一个很好的结合，另外不同的是如下图，
DeepFM的wide和deep部分共享embedding向量空间，wide和deep均可以更新embedding部分
"""

# %load DeepFM.py
#!/usr/bin/env python
"""
#1 Input pipline using Dataset high level API, Support parallel and prefetch reading
#2 Train pipline using Coustom Estimator by rewriting model_fn
#3 Support distincted training using TF_CONFIG
#4 Support export_model for TensorFlow Serving
方便迁移到其他算法上，只要修改input_fn and model_fn
by lambdaji
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time
#import gc
#from multiprocessing import Process

#import math
import random
import pandas as pd
import numpy as np
import tensorflow as tf

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

#1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        #columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        #features = dict(zip(CSV_COLUMNS, columns))
        #labels = features.pop(LABEL_COLUMN)
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        #feat_ids = tf.reshape(feat_ids,shape=[-1,FLAGS.field_size])
        #for i in range(splits.dense_shape.eval()[0]):
        #    feat_ids.append(tf.string_to_number(splits.values[2*i], out_type=tf.int32))
        #    feat_vals.append(tf.string_to_number(splits.values[2*i+1]))
        #return tf.reshape(feat_ids,shape=[-1,field_size]), tf.reshape(feat_vals,shape=[-1,field_size]), labels
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    #return dataset.make_one_shot_iterator()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    #------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    #batch_norm_decay = params["batch_norm_decay"]
    #optimizer = params["optimizer"]
    layers = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))

    #------bulid weights------
    FM_B = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    FM_W = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    # F
    FM_V = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size], initializer=tf.glorot_normal_initializer())
    # F * E    
    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size]) # None * f/K * K
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size]) # None * f/K * K

    #------build f(x)------
    with tf.variable_scope("First-order"):
        feat_wgts = tf.nn.embedding_lookup(FM_W, feat_ids) # None * f/K * K
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1) 

    with tf.variable_scope("Second-order"):
        embeddings = tf.nn.embedding_lookup(FM_V, feat_ids) # None * f/K * K * E
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1]) # None * f/K * K * 1 ？
        embeddings = tf.multiply(embeddings, feat_vals) #vij*xi  
        sum_square = tf.square(tf.reduce_sum(embeddings,1)) # None * K * E
        square_sum = tf.reduce_sum(tf.square(embeddings),1)
        y_v = 0.5*tf.reduce_sum(tf.subtract(sum_square, square_sum),1)	# None * 1

    with tf.variable_scope("Deep-part"):
        if FLAGS.batch_norm:
            #normalizer_fn = tf.contrib.layers.batch_norm
            #normalizer_fn = tf.layers.batch_normalization
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': True, 'reuse': None}
            else:
                train_phase = False
                #normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True, 'updates_collections': None, 'is_training': False, 'reuse': True}
        else:
            normalizer_fn = None
            normalizer_params = None

        deep_inputs = tf.reshape(embeddings,shape=[-1,field_size*embedding_size]) # None * (F*K)
        for i in range(len(layers)):
            #if FLAGS.batch_norm:
            #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
                #normalizer_params.update({'scope': 'bn_%d' %i})
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i],                 #normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, \
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)
            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                #deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,                 weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')
        y_d = tf.reshape(y_deep,shape=[-1])
        #sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]], initializer=tf.glorot_normal_initializer())
        #sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        #deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

    with tf.variable_scope("DeepFM-out"):
        #y_bias = FM_B * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
        y_bias = FM_B * tf.ones_like(y_d, dtype=tf.float32)     # None * 1
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) +         l2_reg * tf.nn.l2_loss(FM_W) +         l2_reg * tf.nn.l2_loss(FM_V) #+ \ l2_reg * tf.nn.l2_loss(sig_wgts)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    #return tf.estimator.EstimatorSpec(
    #        mode=mode,
    #        loss=loss,
    #        train_op=train_op,
    #        predictions={"prob": pred},
    #        eval_metric_ops=eval_metric_ops)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def set_dist_env():
    if FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    tr_files = glob.glob("%s/tr*libsvm" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*libsvm" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*libsvm" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(DeepFM, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        DeepFM.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = DeepFM.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        #feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        #feature_spec = {
        #    'feat_ids': tf.FixedLenFeature(dtype=tf.int64, shape=[None, FLAGS.field_size]),
        #    'feat_vals': tf.FixedLenFeature(dtype=tf.float32, shape=[None, FLAGS.field_size])
        #}
        #serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        DeepFM.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    #------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    #FLAGS.data_dir  = FLAGS.data_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('l2_reg ', FLAGS.l2_reg)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()




# get_ipython().system(u'python DeepFM.py   --task_type=train      --learning_rate=0.0005     --optimizer=Adam  --num_epochs=1                     --batch_size=256                     --field_size=39                     --feature_size=117581                     --deep_layers=400,400,400                     --dropout=0.5,0.5,0.5                     --log_steps=1000                     --num_threads=8                     --model_dir=./criteo_model/DeepFM                     --data_dir=./criteo_data')


# ### NFM
# reference：<br>[从ctr预估问题看看f(x)设计—DNN篇](https://zhuanlan.zhihu.com/p/28202287)<br> [深度学习在CTR预估中的应用](https://zhuanlan.zhihu.com/p/35484389)
# 
# NFM = LR + Embedding + Bi-Interaction Pooling + MLP
# 
# ![](https://pic4.zhimg.com/v2-582ade4feb65a88b828942c460e08192_r.jpg)
# 
# 对不同特征做相同维数的embedding向量。接下来，这些embedding向量两两做element-wise的相乘运算得到B-interaction layer。(element-wide运算举例: (1,2,3)element-wide相乘(4,5,6)结果是(4,10,18)。)
# 
# B-interaction Layer 得到的是一个和embedding维数相同的向量。然后后面接几个隐藏层输出结果。
# 
# 大家思考一下，如果B-interaction layer后面不接隐藏层，直接把向量的元素相加输出结果, 就是一个FM，现在后面增加了隐藏层，相当于做了更高阶的FM，更加增强了非线性表达能力。
# 
# NFM 在 embedding 做了 bi-interaction 操作来做特征的交叉处理，优点是网络参数从 n 直接压缩到 k（比 FNN 和 DeepFM 的 f\*k 还少），降低了网络复杂度，能够加速网络的训练得到模型；但同时这种方法也可能带来较大的信息损失。

# In[ ]:


# %load NFM.py
#!/usr/bin/env python
"""
TensorFlow Implementation of <<Neural Factorization Machines for Sparse Predictive Analytics>> with the fellowing features：
#1 Input pipline using Dataset high level API, Support parallel and prefetch reading
#2 Train pipline using Coustom Estimator by rewriting model_fn
#3 Support distincted training by TF_CONFIG
#4 Support export model for TensorFlow Serving

by lambdaji
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

#import argparse
import shutil
#import sys
import os
import json
import glob
from datetime import date, timedelta
from time import time
#import gc
#from multiprocessing import Process

#import math
import random
import pandas as pd
import numpy as np
import tensorflow as tf

#################### CMD Arguments ####################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 0, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 64, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 128, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.05, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.001, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.8,0.8', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")

#1 1:0.5 2:0.03519 3:1 4:0.02567 7:0.03708 8:0.01705 9:0.06296 10:0.18185 11:0.02497 12:1 14:0.02565 15:0.03267 17:0.0247 18:0.03158 20:1 22:1 23:0.13169 24:0.02933 27:0.18159 31:0.0177 34:0.02888 38:1 51:1 63:1 132:1 164:1 236:1
def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=False):
    print('Parsing', filenames)
    def decode_libsvm(line):
        #columns = tf.decode_csv(value, record_defaults=CSV_COLUMN_DEFAULTS)
        #features = dict(zip(CSV_COLUMNS, columns))
        #labels = features.pop(LABEL_COLUMN)
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values,splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals,num_or_size_splits=2,axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)    # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size) # Batch size to use

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    #return tf.reshape(batch_ids,shape=[-1,field_size]), tf.reshape(batch_vals,shape=[-1,field_size]), batch_labels
    return batch_features, batch_labels

def model_fn(features, labels, mode, params):
    """Bulid Model function f(x) for Estimator."""
    #------hyperparameters----
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    #optimizer = params["optimizer"]
    layers = map(int, params["deep_layers"].split(','))
    dropout = map(float, params["dropout"].split(','))

    #------bulid weights------
    Global_Bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
    Feat_Bias = tf.get_variable(name='linear', shape=[feature_size], initializer=tf.glorot_normal_initializer())
    Feat_Emb = tf.get_variable(name='emb', shape=[feature_size,embedding_size], initializer=tf.glorot_normal_initializer())

    #------build feaure-------
    feat_ids  = features['feat_ids']
    feat_ids = tf.reshape(feat_ids,shape=[-1,field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals,shape=[-1,field_size])

    #------build f(x)------
    with tf.variable_scope("Linear-part"):
        feat_wgts = tf.nn.embedding_lookup(Feat_Bias, feat_ids) 		# None * F * 1
        y_linear = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals),1)

    with tf.variable_scope("BiInter-part"):
        embeddings = tf.nn.embedding_lookup(Feat_Emb, feat_ids) 		# None * F * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals) 				# vij * xi
        sum_square_emb = tf.square(tf.reduce_sum(embeddings,1))
        square_sum_emb = tf.reduce_sum(tf.square(embeddings),1)
        deep_inputs = 0.5*tf.subtract(sum_square_emb, square_sum_emb)	# None * K

    with tf.variable_scope("Deep-part"):
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_phase = True
        else:
            train_phase = False

        if mode == tf.estimator.ModeKeys.TRAIN:
            deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[0]) 						# None * K
        for i in range(len(layers)):
            deep_inputs = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=layers[i],                 weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)

            if FLAGS.batch_norm:
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)   #放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])                              #Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
                #deep_inputs = tf.layers.dropout(inputs=deep_inputs, rate=dropout[i], training=mode == tf.estimator.ModeKeys.TRAIN)

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.identity,             weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='deep_out')
        y_d = tf.reshape(y_deep,shape=[-1])

    with tf.variable_scope("NFM-out"):
        #y_bias = Global_Bias * tf.ones_like(labels, dtype=tf.float32)  # None * 1  warning;这里不能用label，否则调用predict/export函数会出错，train/evaluate正常；初步判断estimator做了优化，用不到label时不传
        y_bias = Global_Bias * tf.ones_like(y_d, dtype=tf.float32)     	# None * 1
        y = y_bias + y_linear + y_d
        pred = tf.sigmoid(y)

    predictions={"prob": pred}
    export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(predictions)}
    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs)

    #------bulid loss------
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=labels)) +         l2_reg * tf.nn.l2_loss(Feat_Bias) + l2_reg * tf.nn.l2_loss(Feat_Emb)

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {
        "auc": tf.metrics.auc(labels, pred)
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops)

    #------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op)

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    #return tf.estimator.EstimatorSpec(
    #        mode=mode,
    #        loss=loss,
    #        train_op=train_op,
    #        predictions={"prob": pred},
    #        eval_metric_ops=eval_metric_ops)

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=True,  reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True, updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z

def set_dist_env():
    if FLAGS.dist_mode == 1:        # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:      # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1] # get first worker as chief
        worker_hosts = worker_hosts[2:] # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index }
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

def main(_):
    tr_files = glob.glob("%s/tr*libsvm" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*libsvm" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*libsvm" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout
    }
    config = tf.estimator.RunConfig().replace(session_config = tf.ConfigProto(device_count={'GPU':0, 'CPU':FLAGS.num_threads}),
            log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    DeepFM = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None, start_delay_secs=1000, throttle_secs=1200)
        tf.estimator.train_and_evaluate(DeepFM, train_spec, eval_spec)
    elif FLAGS.task_type == 'eval':
        DeepFM.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = DeepFM.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size), predict_keys="prob")
        with open(FLAGS.data_dir+"/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\n" % (prob['prob']))
    elif FLAGS.task_type == 'export':
        #feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        #feature_spec = {
        #    'feat_ids': tf.FixedLenFeature(dtype=tf.int64, shape=[None, FLAGS.field_size]),
        #    'feat_vals': tf.FixedLenFeature(dtype=tf.float32, shape=[None, FLAGS.field_size])
        #}
        #serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        DeepFM.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)

if __name__ == "__main__":
    #------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir
    #FLAGS.data_dir  = FLAGS.data_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()


# In[ ]:


# get_ipython().system(u'python NFM.py   --task_type=train                     --learning_rate=0.0005                     --optimizer=Adam                     --num_epochs=1                     --batch_size=256                     --field_size=39                     --feature_size=117581                     --deep_layers=400,400,400                     --dropout=0.5,0.5,0.5                     --log_steps=1000                     --num_threads=8                     --model_dir=./criteo_model/NFM                     --data_dir=./criteo_data')


# ### DeepCTR
# 充分利用图像带来的视觉影响，结合图像信息(通过CNN抽取)和业务特征一起判断点击率大小
# ![](https://pic3.zhimg.com/v2-df0ed2332c6fb09786dfd29a3311b47c_r.jpg)

# In[ ]:


# %load train_with_googlenet.py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
# from keras.layers import Embedding, Merge
from keras.layers import Input,merge,Conv2D,MaxPooling2D,UpSampling2D,Dropout,Embedding

from keras.callbacks import ModelCheckpoint
import keras
from keras.preprocessing import image
import numpy as np
import sys, os, re
from keras.applications.inception_v3 import InceptionV3, preprocess_input

#定义VGG卷积神经网络
def GoogleInceptionV3():
    model = InceptionV3(weights='imagenet', include_top=False)
    model.trainable = False

    return model

#加载field和feature信息
def load_field_feature_meta(field_info_file):
    field_feature_dic = {}
    for line in open(field_info_file):
        contents = line.strip().split("\t")
        field_id = int(contents[1])
        feature_count = int(contents[4])
        field_feature_dic[field_id] = feature_count
    return field_feature_dic

#CTR特征做embedding
def CTR_embedding(field_feature_dic):
    emd = []
    for field_id in range(len(field_feature_dic)):
        # 先把离散特征embedding到稠密的层
        tmp_model = Sequential()
        #留一个位置给rare
        input_dims = field_feature_dic[field_id]+1
        if input_dims>16:
                dense_dim = 16
        else:
                dense_dim = input_dims
        tmp_model.add(Dense(dense_dim, input_dim=input_dims))
        emd.append(tmp_model)
    return emd

#总的网络结构
def full_network(field_feature_dic):
    print ("GoogleNet model loading")
    googleNet_model = GoogleInceptionV3()
    image_model = Flatten()(googleNet_model.outputs)
    image_model = Dense(256)(image_model)
    
    print ("GoogleNet model loaded")
    print ("initialize embedding model")
    print ("loading fields info...")
    emd = CTR_embedding(field_feature_dic)
    print ("embedding model done!")
    print ("initialize full model...")
    full_model = Sequential()
    full_input = [image_model] + emd
    full_model.add(merge(full_input, mode='concat'))
    #批规范化
    full_model.add(keras.layers.normalization.BatchNormalization())
    #全连接层
    full_model.add(Dense(128))
    full_model.add(Dropout(0.4))
    full_model.add(Activation('relu'))
    #全连接层
    full_model.add(Dense(128))
    full_model.add(Dropout(0.4))
    #最后的分类
    full_model.add(Dense(1))
    full_model.add(Activation('sigmoid'))
    #编译整个模型
    full_model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['binary_accuracy','fmeasure'])
    #输出模型每一层的信息
    full_model.summary()
    return full_model


#图像预处理
def vgg_image_preoprocessing(image):
    img = image.load_img(image, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

#CTR特征预处理
def ctr_feature_preprocessing(field_feature_string):
    contents = field_feature_string.strip().split(" ")
    feature_dic = {}
    for content in contents:
        field_id, feature_id, num = content.split(":")
        feature_dic[int(field_id)] = int(feature_id)
    return feature_dic

#产出用于训练的一个batch数据
def generate_batch_from_file(in_f, field_feature_dic, batch_num, skip_lines=0):
    #初始化x和y
    img_x = []
    x = []
    for field_id in range(len(field_feature_dic)):
            x.append(np.zeros((batch_num, int(field_feature_dic[field_id])+1)))
    y = [0.0]*batch_num
    round_num = 1

    while True:
        line_count = 0
        skips = 0
        f = open(in_f)
        for line in f:
            if(skip_lines>0 and round_num==1):
                if skips < skip_lines:
                    skips += 1
                    continue
            if (line_count+1)%batch_num == 0:
                contents = line.strip().split("\t")
                img_name = "images/"+re.sub(r'.jpg.*', '.jpg', contents[1].split("/")[-1])
                if not os.path.isfile(img_name):
                    continue
                #初始化最后一个样本
                try:
                    img_input = vgg_image_preoprocessing(img_name)
                except:
                    continue
                #图片特征填充
                img_x.append(img_input)
                #ctr特征填充
                ctr_feature_dic = ctr_feature_preprocessing(contents[2])
                for field_id in ctr_feature_dic:
                    x[field_id][line_count][ctr_feature_dic[field_id]] = 1.0
                #填充y值
                y[line_count] = int(contents[0])
                #print "shape is", np.array(img_x).shape
                yield ([np.array(img_x)]+x, y)

                img_x = []
                x = []
                for field_id in range(len(field_feature_dic)):
                    x.append(np.zeros((batch_num, int(field_feature_dic[field_id])+1)))
                y = [0.0]*batch_num
                line_count = 0
            else:   
                contents = line.strip().split("\t")
                img_name = "images/"+re.sub(r'.jpg.*', '.jpg', contents[1].split("/")[-1])
                if not os.path.isfile(img_name):
                    continue
                try:
                    img_input = vgg_image_preoprocessing(img_name)
                except:
                    continue
                #图片特征填充
                img_x.append(img_input)
                #ctr特征填充
                ctr_feature_dic = ctr_feature_preprocessing(contents[2])
                for field_id in ctr_feature_dic:
                    x[field_id][line_count][ctr_feature_dic[field_id]] = 1.0
                #填充y值
                y[line_count] = int(contents[0])
                line_count += 1
        f.close()
        round_num += 1

def train_network(skip_lines, batch_num, field_info_file, data_file, weight_file):
    print ("starting train whole network...\n")
    field_feature_dic = load_field_feature_meta(field_info_file)
    full_model = full_network(field_feature_dic)
    if os.path.isfile(weight_file):
        full_model.load_weights(weight_file)
    checkpointer = ModelCheckpoint(filepath=weight_file, save_best_only=False, verbose=1, period=3)
    full_model.fit_generator(generate_batch_from_file(data_file, field_feature_dic, batch_num, skip_lines),samples_per_epoch=1280, nb_epoch=100000, callbacks=[checkpointer])

if __name__ == '__main__':
    skip_lines = sys.argv[1]
    batch_num = sys.argv[2]
    field_info_file = sys.argv[3]
    data_file = sys.argv[4]
    weight_file = sys.argv[5]
    train_network(int(skip_lines), int(batch_num), field_info_file, data_file, weight_file)

