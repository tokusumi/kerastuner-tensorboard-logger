import tensorflow as tf
from tensorboard.plugins.hparams import api as hp_board
from kerastuner.engine.base_tuner import BaseTuner
import kerastuner.engine.hyperparameters as hp

from kerastuner_tensorboard_logger import TensorBoardLogger


def setup_tb(tuner: BaseTuner):
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


def kerastuner_to_hparams(value: hp.HyperParameter) -> hp_board.HParam:
    """convert kerastuner hp to tensorboard hp"""
    SHORTCUT = {
        hp.Int: int_to_Hparam,
        hp.Float: float_to_Hparam,
        hp.Choice: choice_to_Hparam,
        hp.Boolean: boolean_to_Hparam,
    }
    hparam = SHORTCUT.get(type(value), to_hparam)(value)
    return hparam


def to_hparam(value: hp.HyperParameter) -> hp_board.HParam:
    """base convertor."""
    name = value.name
    return hp_board.HParam(name)


def choice_to_Hparam(value: hp.Choice) -> hp_board.HParam:
    """Choice to hp_board.HParam"""
    name = value.name
    choices = value.values
    return hp_board.HParam(name, hp_board.Discrete(choices))


def int_to_Hparam(value: hp.Int) -> hp_board.HParam:
    """Int to hp_board.Hparam"""
    name = value.name
    min_value = value.min_value
    max_value = value.max_value
    return hp_board.HParam(
        name, hp_board.IntInterval(min_value=min_value, max_value=max_value)
    )


def float_to_Hparam(value: hp.Float) -> hp_board.HParam:
    """Float to hp_board.Hparam"""
    name = value.name
    min_value = value.min_value
    max_value = value.max_value
    return hp_board.HParam(
        name, hp_board.RealInterval(min_value=min_value, max_value=max_value)
    )


def boolean_to_Hparam(value: hp.Float) -> hp_board.HParam:
    """boolean to hp_board.Hparam"""
    name = value.name
    hp_board.Discrete
    return hp_board.HParam(name, hp_board.Discrete((True, False)))
