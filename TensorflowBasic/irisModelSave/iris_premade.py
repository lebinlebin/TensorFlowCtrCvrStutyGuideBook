import os
import tensorflow as tf
import shutil
import TensorflowBasic.irisModelSave.iris_load as dts

#利用iris_load.py读取训练数据和测试数据
(train_x, train_y), (test_x, test_y) = dts.load()

print (train_x)
print(">>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<")
print (train_y)

#设定特征值的名称
feature_columns = []
for key in train_x:
    feature_columns.append(tf.feature_column.numeric_column(key=key))   

#估计器存储路径
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
models_path=os.path.join(dir_path,'models/')

#估计器存储设置选项
ckpt_config= tf.estimator.RunConfig(
    save_checkpoints_secs = 60,  #每60秒保存一次
    keep_checkpoint_max = 10,    #保留最近的10次
)

#估计器预设
my_cfg=dict() 
my_cfg['layer1'],my_cfg['layer2'],my_cfg['batch_size'],my_cfg['steps']=10,10,100,1000
"""
checkpoints和SavedModel
Tensorflow可以将训练好的模型以两种形式保存：
1.	chekpoints检查点集，依赖于创建模型的代码
2.	SavedModel已保存模型，不依赖于创建模型的代码
修改iris.py文件中创建评估器/分类器的代码，添加model_dir保存模型的目录：
#选定估算器：深层神经网络分类器
models_path=os.path.join(dir_path,'models/')
classifier = tf.estimator.DNNClassifier(
   feature_columns=feature_columns,
   hidden_units=[10, 10],
   n_classes=3,
   model_dir=models_path)
"""
#生产估计器函数：深层神经网络分类器
def estimator():
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[my_cfg['layer1'], my_cfg['layer2']],
        n_classes=3,
        model_dir=models_path,#
        config=ckpt_config)
    return classifier 

#训练模型函数
def train():
    print('Please input:layer1 nodes,layer2 nodes,batch_size,steps')
    params=input().split(',')
    if len(params)>3:
        if os.path.exists(models_path):
            print('Removing models folder...')
            shutil.rmtree(models_path) #移除models目录
            
        my_cfg['layer1'],my_cfg['layer2'],my_cfg['batch_size'],my_cfg['steps'] = map(int, params)
        
    print('Training...')
    classifier=estimator()   
    classifier.train(input_fn=lambda:dts.train_input_fn(
            train_x,
            train_y,
            my_cfg['batch_size']),
         steps=my_cfg['steps'])
    print('Train OK')         
        

#评估模型函数
def evalute():
    print('Evaluating...') 
    classifier=estimator()  
    eval_result = classifier.evaluate(
        input_fn=lambda:dts.eval_input_fn(test_x, test_y,my_cfg['batch_size']))
    print('Evaluate result:',eval_result)
    
def predict():
    print('Please enter features: SepalLength,SepalWidth,PetalLength,PetalWidth;0 for exit.')
    params=input().split(',');
    if len(params)>3:
        predict_x = {
            'SepalLength': [float(params[0])],
            'SepalWidth': [float(params[1])],
            'PetalLength': [float(params[2])],
            'PetalWidth': [float(params[3])],
        }    

        #进行预测
        classifier=estimator()        
        predictions = classifier.predict(
                input_fn=lambda:dts.eval_input_fn(predict_x,
                                                labels=[0],
                                                batch_size=my_cfg['batch_size']))

        #预测结果是数组，尽管实际我们只有一个
        for pred_dict in predictions:
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            print('Predict result:',dts.SPECIES[class_id],100 * probability)
    else:
        print('Input format error,ignored.')

#定义入口主函数
def main(args):
    while 1==1:
        print('Please enter train,evalute or predict:')
        cmd = input() #捕获用户输入的数字
        if cmd=='train':
            train()
        elif cmd=='evalute':
            evalute()
        elif cmd=='predict':
            predict()
        # elif cmd=='retrain':
        #     retrain()
            
#运行主函数            
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)