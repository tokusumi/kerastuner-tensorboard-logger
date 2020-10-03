import pytest

from tensorboard.plugins.hparams import api as hp_board
import kerastuner.engine.hyperparameters as hp

from kerastuner_tensorboard_logger.initialize import kerastuner_to_hparams


def test_kerastuner_to_hparams():
    name = "not support"
    value = hp.HyperParameter(name)
    hparam = kerastuner_to_hparams(value)
    assert hparam.name == name
    assert not hparam.domain

    name = "choice"
    value = hp.Choice(name, (1, 2, 3))
    hparam = kerastuner_to_hparams(value)
    assert hparam.name == name
    assert isinstance(hparam.domain, hp_board.Discrete)

    name = "int"
    value = hp.Int(name, 0, 10)
    hparam = kerastuner_to_hparams(value)
    assert hparam.name == name
    assert isinstance(hparam.domain, hp_board.IntInterval)

    name = "float"
    value = hp.Float(name, 0, 1, step=0.1)
    hparam = kerastuner_to_hparams(value)
    assert hparam.name == name
    assert isinstance(hparam.domain, hp_board.RealInterval)

    name = "boolean"
    value = hp.Boolean(name)
    hparam = kerastuner_to_hparams(value)
    assert hparam.name == name
    assert isinstance(hparam.domain, hp_board.Discrete)
