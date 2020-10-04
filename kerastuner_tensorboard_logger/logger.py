import os
from typing import Any, Dict, Generator, List, Tuple, Union
from datetime import timedelta, datetime
import tensorflow as tf
from kerastuner.engine.logger import Logger
from tensorboard.plugins.hparams import api as hp_board


def timedelta_to_hms(timedelta: timedelta) -> str:
    """convert datetime.timedelta to string like '01h:15m:30s'"""
    tot_seconds = int(timedelta.total_seconds())
    hours = tot_seconds // 3600
    minutes = (tot_seconds % 3600) // 60
    seconds = tot_seconds % 60
    return f"{hours}h{minutes}m{seconds}s"


class TensorBoardLogger(Logger):
    """tensorboard.plugins.hparams logger class.

    Create logger instance for tuner instance that inherit kerastuner.engine.base_tuner.BaseTuner.

    # Arguments:
        metrics: String or list of String, reports values for Tensorboard.
        logdir: String. Path to the log directory (relative) to be accessed by Tensorboard.
        overwrite: Bool, default `False`. If `False`, reloads an existing log directory
            of the same name if one is found. Otherwise, overwrites the log.
    """

    def __init__(
        self,
        metrics: Union[str, List[str]] = ["acc"],
        logdir: str = "logs/hparam_tuning",
        overwrite: bool = False,
    ):
        self.metrics = [metrics] if isinstance(metrics, str) else metrics
        self.logdir = logdir
        self.times: Dict[str, datetime] = dict()

        if overwrite and tf.io.gfile.exists(self.logdir):
            tf.io.gfile.rmtree(self.logdir)

    def register_tuner(self, tuner_state):
        """Informs the logger that a new search is starting."""
        pass

    def register_trial(self, trial_id: str, trial_state: Dict[str, Any]):
        """Informs the logger that a new Trial is starting."""
        self.times[trial_id] = datetime.now()

    def report_trial_state(self, trial_id: str, trial_state: Dict[str, Any]):
        """Gives the logger information about trial status."""
        execution_time = timedelta_to_hms(datetime.now() - self.times.pop(trial_id))
        name = f"{execution_time}-{trial_id}"
        logdir = os.path.join(self.logdir, name)

        with tf.summary.create_file_writer(logdir).as_default():
            hparams = self.parse_hparams(trial_state)
            hp_board.hparams(
                hparams, trial_id=name
            )  # record the values used in this trial

            for target_metric, metric in self.parse_metrics(trial_state):
                tf.summary.scalar(target_metric, metric, step=1)

    def exit(self):
        pass

    def parse_hparams(self, trial_state: Dict[str, Any]) -> Dict[str, Any]:
        """use in tf.summary writer context"""
        hparams = trial_state.get("hyperparameters", {}).get("values", {})
        return hparams

    def parse_metrics(
        self, trial_state: Dict[str, Any]
    ) -> Generator[Tuple[str, Union[int, float]], None, None]:
        """use in tf.summary writer context"""
        metrics = trial_state.get("metrics", {}).get("metrics", {})
        for target_metric in self.metrics:
            metric = metrics.get(target_metric)
            if metric is None:
                continue
            metric = metric.get("observations", [{}])[0].get("value")
            if metric is None:
                continue
            yield target_metric, metric[0]
