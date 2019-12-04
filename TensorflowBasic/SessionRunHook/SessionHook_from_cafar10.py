import tensorflow as tf
import  time
class _LoggerHook(tf.train.SessionRunHook):
  """Logs loss and runtime."""

  def begin(self):
    self._step = -1
    self._start_time = time.time()

  def before_run(self, run_context):
    self._step += 1
    return tf.train.SessionRunArgs(loss)  # Asks for loss value.

  def after_run(self, run_context, run_values):
    if self._step % FLAGS.log_frequency == 0:
      current_time = time.time()
      duration = current_time - self._start_time#duration持续的时间
      self._start_time = current_time

      loss_value = run_values.results
      examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
      sec_per_batch = float(duration / FLAGS.log_frequency)

      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print (format_str % (datetime.now(), self._step, loss_value,
                           examples_per_sec, sec_per_batch))


#monitored 被监控的
with tf.train.MonitoredTrainingSession(
    checkpoint_dir=FLAGS.train_dir,
    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
           tf.train.NanTensorHook(loss),
           _LoggerHook()],
    config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement)) as mon_sess:
  while not mon_sess.should_stop():
    mon_sess.run(train_op)

