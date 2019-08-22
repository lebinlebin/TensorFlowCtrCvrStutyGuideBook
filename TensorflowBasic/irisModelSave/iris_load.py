import os
import pandas as pd
import tensorflow as tf

FUTURES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    
#格式化数据文件的目录地址
dir_path = os.path.dirname(os.path.realpath(__file__))
train_path=os.path.join(dir_path,'iris_training.csv')
test_path=os.path.join(dir_path,'iris_test.csv')
    
#载入数据函数
def load():    
    #载入训练数据
    train = pd.read_csv(train_path, names=FUTURES, header=0)
    train_x, train_y = train, train.pop('Species')

    #载入测试数据
    test = pd.read_csv(test_path, names=FUTURES, header=0)
    test_x, test_y = test, test.pop('Species') 
    
    return (train_x, train_y),(test_x, test_y)    
    
#针对训练的喂食函数
def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size) #每次随机调整数据顺序
    return dataset


#针对测试的喂食函数
def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    inputs=(features,labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    return dataset
