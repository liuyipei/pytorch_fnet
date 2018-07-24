# Tensorflow reimplementation of fnet_model
# Assumes that the tfrecrods have already been generated separately
#
# python3 fnet/tf_estimator_wrap.py
# python3 fnet/tf_estimator_wrap.py --learning_rate=0.001 > run_adam.out 2> run_adam.err
# python3 fnet/tf_estimator_wrap.py --learning_rate=0.001 --model_dir=tf_checkpoints/lessoutof > run_lessoutof.out 2> run_lessoutof.err
# python3 fnet/tf_estimator_wrap.py --learning_rate=0.001 --model_dir=tf_checkpoints/xavierfix > run_xavierfix.out 2> run_xavierfix.err

import tensorflow as tf
import numpy as np
import pytorch_fnet.fnet.tf_model as tf_model
import subprocess
# import tempfile
import absl.app as app
import absl.flags as flags
import absl.logging as logging


FLAGS = flags.FLAGS


flags.DEFINE_integer('batch_size', 24, 'batch size')
flags.DEFINE_integer('steps', 100000, 'steps')
flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
flags.DEFINE_bool('tpuspec', False, 'tpuspec')
flags.DEFINE_integer('num_patches', 100, 'patches taken per image in tf.dataset')
flags.DEFINE_bool('multi_patch', True, 'crop multiple matches per image read in tf.dataset')
flags.DEFINE_string('model_dir', 'tf_checkpoints/adam_patches', 'model_dir')
flags.DEFINE_string('tfrecord_input',  'tfrecords/dna-train.tfrecord', 'tfrecord_input')
flags.DEFINE_integer('tpu_iterations', 3, 'tpu_iterations')
flags.DEFINE_integer('tpu_num_shards', 8, 'tpu_num_shards')
flags.DEFINE_integer('tpu_name', None, 'tpu_name')
patch_zyx = [32, 64, 64]  # TODO make this flags


def _parse_function(example_proto, patch_zyx, multi_patch=False, num_patches=20):
    features = {"train/signal": tf.VarLenFeature(tf.string), # float32 numpy array 
                "train/target": tf.VarLenFeature(tf.string), # float32 numpy array
                "train/shape": tf.VarLenFeature(tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)

    full_shape = tf.sparse_tensor_to_dense(parsed_features['train/shape'])
    full_shape.set_shape([3,])
    assert_op = tf.assert_equal(tf.shape(full_shape), np.array([3], dtype=np.int32))

    signal_decoded = tf.decode_raw(tf.sparse_tensor_to_dense(parsed_features['train/signal'], default_value=chr(0)), tf.float32)
    target_decoded = tf.decode_raw(tf.sparse_tensor_to_dense(parsed_features['train/target'], default_value=chr(0)), tf.float32)
    with tf.control_dependencies([assert_op]):
        signal_reshaped = tf.reshape(signal_decoded, full_shape)
        target_reshaped = tf.reshape(target_decoded, full_shape)
        signal_reshaped.set_shape([None, None, None])
        target_reshaped.set_shape([None, None, None])
        sigtar_2ZHW = tf.stack([signal_reshaped, target_reshaped])
    sigtar_crop_shape = [2, ]
    sigtar_crop_shape.extend(patch_zyx) 
    if not multi_patch:
        sigtar_2zhw = tf.random_crop(sigtar_2ZHW, sigtar_crop_shape)
        return sigtar_2zhw, full_shape
    else:
        sigtar_B2zhw = tf.stack([tf.random_crop(sigtar_2ZHW, sigtar_crop_shape) for _ in range(num_patches)], axis=0)
        full_shape_B3 = tf.stack([full_shape for _ in range(num_patches)], axis=0)
        return sigtar_B2zhw, full_shape_B3


def allen_input_fn(filenames, patch_zyx, batch_size=24, multi_patch=True, num_patches=20):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.shuffle(batch_size) # heuristic buffer size
    dataset = dataset.map(lambda e: _parse_function(e, patch_zyx, multi_patch=multi_patch, num_patches=num_patches), num_parallel_calls=4)
    dataset = dataset.apply(tf.contrib.data.unbatch())
    patch_buffer_size = num_patches * batch_size if multi_patch else batch_size * 2 # heuristic on both sides 
    dataset = dataset.shuffle(patch_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    iterator = dataset.make_one_shot_iterator()
    sigtar_N2DHWC, dhw_shape_N3 = iterator.get_next()
    return sigtar_N2DHWC, dhw_shape_N3


def allen_model_fn(
    features, # This is from input_fn
    labels,   # This is from input_fn
    mode,
    params):    # And instance of tf.estimator.ModeKeys, see below
    tpuspec = params.get('tpuspec', False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    sigtar_N2DHWC = features
    signal_NDHW1 = tf.expand_dims(sigtar_N2DHWC[:, 0, :, :, :], axis=-1)
    target_NDHW1 = tf.expand_dims(sigtar_N2DHWC[:, 1, :, :, :], axis=-1)

    var_dict = {}
    pred_NDHW1 = tf_model.allen_net(signal_NDHW1, var_dict, training=is_training)
    predictions = {'semantic': pred_NDHW1}

    # 1. Prediction mode
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluation and Training mode

    # Calculate the loss
    if tpuspec:
        def loss_fn(pred, target):
            return tf.reduce_mean((pred - target) ** 2)
    loss = tf.reduce_mean((pred_NDHW1 - target_NDHW1) ** 2)

    # Calculate the accuracy between the true labels, and our predictions
    # accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

    # 2. Evaluation mode
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accurace to the accuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar

    if mode == tf.estimator.ModeKeys.EVAL:
        if tpuspec:
            return tf.contrib.tpu.TPUEstimatorSpec(
                mode,
                loss=loss,
                eval_metrics=(loss_fn, [pred_NDHW1, target_NDHW1]))
        else:
            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                eval_metric_ops={'eval-loss': loss})

    # If mode is not PREDICT nor EVAL, then we must be in TRAIN
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    # 3. Training mode    
    # torch code ## self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.5, 0.999))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'], beta1=0.5, beta2=0.999) # default is (.9, 0.999)
    if tpuspec:
        optimizer = tf.contrib.tpu.TPUEstimator(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    # init_op = tf.global_variables_initializer()
    # optimizer = tf.train.AdagradOptimizer(0.05)
    # train_op = optimizer.minimize(
    #    loss,
    #    global_step=tf.train.get_global_step())

    # Set the TensorBoard scalar my_accuracy to the accuracy
    # Obs: This function only sets the value during mode == ModeKeys.TRAIN
    # To set values during evaluation, see eval_metrics_ops
    if not tpuspec:
        tf.summary.scalar('loss', loss)

    # Return training operations: loss and train_op
    if tpuspec:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)
    else:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)


def main(argv):
    del argv  # Unused.
    print(FLAGS)
    params = dict()
    params['tpuspec'] = FLAGS.tpuspec
    params['learning_rate'] = FLAGS.learning_rate
    if FLAGS.tpuspec:
        # copy paste code from public example; preparing for google3

        my_project_name = subprocess.check_output([
            'gcloud','config','get-value','project'])
        my_zone = subprocess.check_output([
            'gcloud','config','get-value','compute/zone'])
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            tpu_names=[FLAGS.tpu_name],
            zone=my_zone,
            project=my_project_name)
        master = tpu_cluster_resolver.get_master()

        my_tpu_run_config = tf.contrib.tpu.RunConfig(
            master=master,
            evaluation_master=master,
            model_dir=FLAGS.model_dir,
            session_config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.tpu_iterations,
                                                FLAGS.tpu_shards),
        )
        my_tpu_estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=allen_model_fn,
            config=my_tpu_run_config,
            use_tpu=False,
            params=params)
        classifier = my_tpu_estimator
    else:
        
        regular_run_config = tf.estimator.RunConfig(
            #model_dir=None,
            #tf_random_seed=None,
            save_summary_steps=1,
            save_checkpoints_steps=10,
            #save_checkpoints_secs=_USE_DEFAULT,
            session_config=tf.ConfigProto(log_device_placement=True),
            keep_checkpoint_max=100,
            #keep_checkpoint_every_n_hours=10000,
            #log_step_count_steps=100,
            #train_distribute=None
        )
        classifier = tf.estimator.Estimator(
            model_fn=allen_model_fn,
            config=regular_run_config,
            model_dir=FLAGS.model_dir,
            params=params)

    if FLAGS.mode == 'train':
        classifier.train(
            input_fn=lambda:allen_input_fn([FLAGS.tfrecord_input], patch_zyx, batch_size=FLAGS.batch_size,
                                           multi_patch=FLAGS.multi_patch, num_patches=FLAGS.num_patches),
                                           steps=FLAGS.steps)
    elif FLAGS.mode == 'eval':
        classifier.evaluate(
            input_fn=lambda:allen_input_fn([FLAGS.tfrecord_input], patch_zyx, batch_size=FLAGS.batch_size,
                                           multi_patch=FLAGS.multi_patch, num_patches=FLAGS.num_patches),
                                           steps=FLAGS.steps)
    else:
        raise ValueError("Unknown mode %s" % FLAGS.mode)

if __name__ == '__main__':
    app.run(main)
