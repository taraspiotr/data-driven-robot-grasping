from typing import Any

from copy import deepcopy

import rlpyt.utils.logging.logger as logger
from mrunner.helpers import client_helper

_rec_tabular = deepcopy(logger.record_tabular)


def record_tabular(key: str, val: Any, *args, **kwargs):
    _rec_tabular(key, val, *args, **kwargs)
    if val != val:
        val = 0
    client_helper.experiment_[key].log(val)


def setup_logger():
    logger.record_tabular = record_tabular
