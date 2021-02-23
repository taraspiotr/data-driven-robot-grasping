from typing import Any

from copy import deepcopy

import neptune
import rlpyt.utils.logging.logger as logger

_rec_tabular = deepcopy(logger.record_tabular)


def record_tabular(key: str, val: Any, *args, **kwargs):
    _rec_tabular(key, val, *args, **kwargs)
    neptune.log_metric(key, val)


def setup_logger():
    logger.record_tabular = record_tabular
