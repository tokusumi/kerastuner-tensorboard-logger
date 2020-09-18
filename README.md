# Keras-tuner Tensorboard logger

keras-tuner logger for streaming search report to Tensorboard plugins Hparams, beautiful interactive visualization tool.

## Requirements

Python 3.6+

## Example

simple and incomplete code is introduced.

See details about how to use keras-tuner in [here](https://github.com/keras-team/keras-tuner).

Add only one argument in tuner class and search it, then you can go to see search report in Tensorboard.

```python
# import this
from kerastuner_tensorboard_logger import TensorBoardLogger

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

tuner.search(x, y, epochs=5, validation_data=(val_x, val_y))
```

### Tensorboard

```bash
$ tensorboard --logdir ./logs/hparams
```

Go to http://127.0.0.1:6006.

You will see the interactive visualization (provided by Tensorboard).

![Table View](https://raw.githubusercontent.com/tokusumi/kerastuner-tensorboard-logger/master/docs/src/table_view.jpg)

![Parallel Coordinates View](https://raw.githubusercontent.com/tokusumi/kerastuner-tensorboard-logger/master/docs/src/parallel_coordinates_view.jpg)