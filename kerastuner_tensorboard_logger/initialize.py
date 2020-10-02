import tensorflow as tf
from tensorboard.plugins.hparams import api as hp_board
from kerastuner.engine.base_tuner import BaseTuner

from kerastuner_tensorboard_logger import TensorBoardLogger


def setup(tuner: BaseTuner):
    """setup hparams"""
    if not isinstance(tuner.logger, TensorBoardLogger):
        raise ValueError("Must set TensorBoardLogger")

    hps = tuner.oracle.get_space()

    hparams = [kerastuner_to_hparams(hp) for hp in hps.space]

    metrics = tuner.logger.metrics
    hmetrics = [hp_board.Metric(metric) for metric in metrics]

    logdir = tuner.logger.logdir
    with tf.summary.create_file_writer(logdir).as_default():
        hp_board.hparams_config(
            hparams=hparams,
            metrics=hmetrics,
        )


def kerastuner_to_hparams(hp):
    return hp