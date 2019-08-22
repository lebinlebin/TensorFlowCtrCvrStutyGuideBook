"""
tensorflow-MonitoredTrainingSession解读
　　MonitoredTrainingSession是tensorflow管理分布式训练中一个重要方法，它相当于集成了一些监控训练组件，
如init、summary、log、save等。在早期的版本，一般使用tf.train.Supervisor来管理session，
后来框架升级后，官方就推荐用MonitoredTrainingSession了。
"""

"""
一、训练为什么要管理？
　　搭建一个简单的分布式训练是不需要管理的，只需要定义好ClusterSpec，给每个节点分配Server，建好图，就可以开始迭代了。最简单的代码如下：
"""
# import tensorflow as tf
# ps_hosts = [xx.xx.xx.xx: xxxx]
# worker_hosts = [xx.xx.xx.xx:xxxx, xx.xx.xx.xx:xxxx]
#
# cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
# server = tf.train.Server(cluster,
#                            job_name=FLAGS.job_name,
#                            task_index=FLAGS.task_index)
# if FLAGS.job_name == "ps":
#     server.join()
#   elif FLAGS.job_name == "worker":
# sess = tf.Session()
# with tf.device(tf.train.replica_device_setter(
#         worker_device="/job:worker/task:%d" % FLAGS.task_index,
#         cluster=cluster)):
# # build_graph()
#     step = 0
#     while step < FLAGS.total_step:
#         sess.run()

"""
随着问题和模型的复杂化，我们也许会有监控训练的需求，如记录日志、训练可视化、checkpoint、early-stop、训练效率调优等，
tensorflow提供了大量的工具支持，但这就加重了代码的复杂度。所以tensorflow封装了MonitoredTrainingSession，将各种监控训练的组件外挂到一个类里.
"""

"""
二、MonitoredTrainingSession参数
"""
# import tensorflow as tf
# tf.train.MonitoredTrainingSession(
#     master='',
#     is_chief=True,
#     checkpoint_dir=None,
#     scaffold=None,
#     hooks=None,
#     chief_only_hooks=None,
#     save_checkpoint_secs=USE_DEFAULT,
#     save_summaries_steps=USE_DEFAULT,
#     save_summaries_secs=USE_DEFAULT,
#     config=None,
#     stop_grace_period_secs=120,
#     log_step_count_steps=100,
#     max_wait_secs=7200,
#     save_checkpoint_steps=USE_DEFAULT,
#     summary_dir=None
# )

"""
　　args:
　　　　master: server.target
　　　　is_chief: 是否为chief（一般把task_index=0定为chief）。chief节点会负责初始化和模型restore，其他节点只需等待chief初始化完成
　　　　checkpoint_dir: checkpoint文件路径
　　　　scaffold：用于完成图表
　　　　hooks：最重要的参数。它是一个SessionRunHook对象的列表，包含了所有希望外挂的组件，如CheckpointSaverHook、FeedFnHook、LoggerTensorHook、NanTensorHook、ProfileHook、StopAtStepHook等，
                也可以自定义Hook，只要继承SessionRunHook类就行。下面会详细介绍几个重要Hook
　　　　chief_only_hooks：只有chief节点才会生效的hook
　　　　save_checkpoint_secs：保存checkpoint的频率
　　　　save_summaries_steps：按步数保存summary的频率 ；save_summaries_secs是按时间
　　　　config：session配置，是ConfigProtoproto格式
　　实例化后就得到一个MonitoredSession对象，可以当作普通session使用

"""

"""
三、Hook的使用
　　Hook顾名思义，是一个“外挂”的组件，用于执行训练中的各种功能。
　　Hook的基类是tf.train.SessionRunHook，需要实现下面几个方法：
"""

"""
1. 
　　after_create_session(
    session,
    coord
)　　
在session被创建后调用


2.
after_run(
    run_context,
    run_values
)
在每次session.run后被调用
3. 
　before_run(run_context)
　每次run前调用
4. 
　begin()
　调用后，图就不能再修改
5. 
　end(session)
　结束session

"""