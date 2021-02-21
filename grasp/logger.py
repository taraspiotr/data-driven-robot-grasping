from copy import deepcopy

import neptune
import rlpyt.utils.logging.logger as logger

_rec_tabular = deepcopy(logger.record_tabular)


def record_tabular(key, val, *args, **kwargs):
    _rec_tabular(key, val, *args, **kwargs)
    # if not _disabled and not _tabular_disabled:
    # key = _tabular_prefix_str + str(key)
    # _tabular.append((key, str(val)))
    # if _tf_summary_writer is not None:
    #     _tf_summary_writer.add_scalar(key, val, _iteration)
    neptune.log_metric(key, val)


def setup_logger():
    logger.record_tabular = record_tabular
