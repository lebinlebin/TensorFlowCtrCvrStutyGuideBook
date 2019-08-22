class SessionRunHook(object):
  """Hook to extend calls to MonitoredSession.run()."""
  def begin(self):
    """在创建会话之前调用
    调用begin()时，default graph会被创建，
    可在此处向default graph增加新op,
    begin()调用后，default graph不能再被修改
    """
    pass
  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    """tf.Session被创建后调用
    调用后会指示所有的Hooks有一个新的会话被创建
    Args:
      session: A TensorFlow Session that has been created.
      coord: A Coordinator object which keeps track of all threads.
    """
    pass
  def before_run(self, run_context):  # pylint: disable=unused-argument
    """调用在每个sess.run()执行之前
    可以返回一个tf.train.SessRunArgs(op/tensor),在即将运行的会话中加入这些op/tensor；
    加入的op/tensor会和sess.run()中已定义的op/tensor合并，然后一起执行；
    Args:
      run_context: A `SessionRunContext` object.
    Returns:
      None or a `SessionRunArgs` object.
    """
    return None
  def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):  # pylint: disable=unused-argument
    """调用在每个sess.run()之后
    参数run_values是befor_run()中要求的op/tensor的返回值；
    可以调用run_context.qeruest_stop()用于停止迭代
    sess.run抛出任何异常after_run不会被调用
    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    pass

  def end(self, session):  # pylint: disable=unused-argument
    """在会话结束时调用
    end()常被用于Hook想要执行最后的操作，如保存最后一个checkpoint
    如果sess.run()抛出除了代表迭代结束的OutOfRange/StopIteration异常外，
    end()不会被调用
    Args:
      session: A TensorFlow Session that will be soon closed.
    """
    pass

"""
tf.train.SessionRunHook()类中定义的方法的参数run_context, run_values, run_args, 包含sess.run()会话运行所需的一切信息，
•	run_context：类tf.train.SessRunContext的实例
•	run_values  ：类tf.train.SessRunValues的实例
•	run_args     ：类tf.train.SessRunArgs的实例
"""


"""
tf.train.SessionRunHook()的使用
（1）可以使用tf中已经预定义好的Hook,其都是tf.train.SessionRunHook()的子类；如
•	StopAtStepHook:设置用于停止迭代的max_step或num_step,两者只能设置其一
•	NanTensorHook:如果loss的值为Nan，则停止训练；
tensorflow中有许多预定义的Hook，想了解更多的同学可以去官方文档tf.train.下查看

（2）也可用tf.train.SessionRunHook()定义自己的Hook,并重写类中的方法；
然后把想要使用的Hook(预定义好的或者自己定义的)放到
1) tf.train.MonitorTrainingSession()参数[Hook]列表中；
2) 加入到  TrainSpec 和  EvalSpec 中作为控制训练的选项
    eg:
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: get_data(traindata_name_reg, epoch_num=None,bz=bz),hooks=train_hook_list)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: get_data(testdata_name_reg, epoch_num=1, bz=eval_bz), steps=1000,
                                      start_delay_secs=60, throttle_secs=300, hooks=eval_hook_list,exporters=export_hook_list)

"""

"""
SessRunContext/SessRunValues/SessRunArgs
这三个类tf.train.SessRunContext/tf.train.SessRunValues/tf.train.SessRunArgs服务于sess.run();
tf.train.SessRunContext/tf.train.SessRunArgs提供会话运行所需的信息，  
tf.train.SessRunValues保存会话运行的结果

      (1)    tf.train.SessRunArgs类
      提供给会话运行的参数，与sess.run()参数定义一样：
       fethes,feeds,option
     (2)    tf.train.SessRunValues
       用于保存sess.run()的结果，
       其中resluts是sess.run()返回值中对应于SessRunArgs()的返回值，

   （3)    tf.train.SessRunContext
       SessRunContext包含sess.run()所需的一切信息
        属性:
•	    original_args:    sess.run所需的参数，是一个tf.train.SessRunArgs实例
•	    session:指定要运行的会话
•	     stop_request:返回一个bool值，用于判断是否停止迭代；
        方法：
•	      request_stop(): 设置_stop_request值为True
"""

