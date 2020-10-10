#!/usr/bin/env bash

set -e

pytest --disable-warnings tests/
tensorboard --logdir tests/logs