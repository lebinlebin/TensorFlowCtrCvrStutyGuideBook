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
DeepCTR
充分利用图像带来的视觉影响，结合图像信息(通过CNN抽取)和业务特征一起判断点击率大小
(https://pic3.zhimg.com/v2-df0ed2332c6fb09786dfd29a3311b47c_r.jpg)
"""
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

