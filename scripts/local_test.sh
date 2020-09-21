#!/usr/bin/env bash

set -e

pytest --disable-warnings tests/
tensorboard --logdir logs/hparam_tuning