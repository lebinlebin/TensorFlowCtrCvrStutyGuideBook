"""
In this article, I will talk about what kinds of things you might want to specify in the train_spec and the eval_spec.
"""
import tensorflow as tf

def train_and_evaluate(output_dir):
    EVAL_INTERVAL = 300 # seconds
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,keep_checkpoint_max = 3)
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = output_dir,
        config = run_config)

    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset('train',
		tf.estimator.ModeKeys.TRAIN, BATCH_SIZE),
        max_steps = TRAIN_STEPS)
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset('eval', tf.estimator.ModeKeys.EVAL, 2**15),  # no need to batch in eval
        steps = None,
        start_delay_secs = 60, # start evaluating after N seconds
        throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

