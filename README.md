# Keras-tuner Tensorboard logger

![](https://github.com/tokusumi/kerastuner-tensorboard-logger/workflows/Tests/badge.svg)
[![PyPI version](https://badge.fury.io/py/kerastuner-tensorboard-logger.svg)](https://badge.fury.io/py/kerastuner-tensorboard-logger)

[keras-tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) logger for streaming search report to [Tensorboard plugins Hparams](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams), beautiful interactive visualization tool.

## Requirements

* Python 3.6+
* keras-tuner 1.0.0+
* Tensorboard 2.1+

## Installation

```
$ pip install kerastuner-tensorboard-logger
```

## Example

here is simple (and incomplete) code.

See details about how to use keras-tuner [here](https://github.com/keras-team/keras-tuner).

Add only one argument in tuner class and search it, then you can go to see search report in Tensorboard.

Optionally, you can call `setup_tb` to be more accurate TensorBoard visualization. It convert keras-tuner hyperparameter information and do [Tensorboard experimental setup](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams#1_experiment_setup_and_the_hparams_experiment_summary).

```python
# import this
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb  # Optional
)

tuner = Hyperband(
    build_model,
    objective="val_acc",
    max_epochs=5,
    directory="logs/tuner",
    project_name="tf_test",
    logger=TensorBoardLogger(
        metrics=["val_acc"], logdir="logs/hparams"
    ),  # add only this argument
)

setup_tb(tuner)  # (Optional) For more accurate visualization.
tuner.search(x, y, epochs=5, validation_data=(val_x, val_y))
```

### Tensorboard

```bash
$ tensorboard --logdir ./logs/hparams
```

Go to http://127.0.0.1:6006.

You will see the interactive visualization (provided by Tensorboard).

![Table View](https://raw.githubusercontent.com/tokusumi/kerastuner-tensorboard-logger/main/docs/src/table_view.jpg)

![Parallel Coordinates View](https://raw.githubusercontent.com/tokusumi/kerastuner-tensorboard-logger/main/docs/src/parallel_coordinates_view.jpg)
