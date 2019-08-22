"""
分布式tensorflow
推荐使用 TensorFlow Estimator API 来编写分布式训练代码，理由如下：
•	开发方便，比起low level的api开发起来更加容易
•	可以方便地和其他的高阶API结合使用，比如Dataset、FeatureColumns、Head等
•	模型函数model_fn的开发可以使用任意的low level函数，依然很灵活
•	单机和分布式代码一致，且不需要考虑底层的硬件设施
•	可以比较方便地和一些分布式调度框架（e.g. xlearning）结合使用
要让tensorflow分布式运行，首先我们需要定义一个由参与分布式计算的机器组成的集群，如下：

"""

cluster = {'chief': ['host0:2222'],
             'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}

"""
集群中一般有多个worker，需要指定其中一个worker为主节点（cheif），chief节点会执行一些额外的工作，比如模型导出之类的。在PS分布式架构环境中，还需要定义ps节点。
要运行分布式Estimator模型，只需要设置好TF_CONFIG环境变量即可，可参考如下代码：
"""
# import os
# # Example of non-chief node:
#   os.environ['TF_CONFIG'] = json.dumps(
#       {'cluster': cluster,
#        'task': {'type': 'worker', 'index': 1}})
#   # Example of chief node:
#   os.environ['TF_CONFIG'] = json.dumps(
#       {'cluster': cluster,
#        'task': {'type': 'chief', 'index': 0}})
#   # Example of evaluator node (evaluator is not part of training cluster)
#   os.environ['TF_CONFIG'] = json.dumps(
#       {'cluster': cluster,
#        'task': {'type': 'evaluator', 'index': 0}})
