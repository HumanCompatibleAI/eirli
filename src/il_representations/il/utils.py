"""Utilities that are helpful for several pieces of IL code (e.g. in both
il_train.py and joint_training.py)."""


def streaming_extract_keys(*keys_to_keep):
    """Filter a generator of dicts to keep only the specified keys."""
    def gen(data_iter):
        for data_dict in data_iter:
            yield {k: data_dict[k] for k in keys_to_keep}

    return gen


def add_infos(data_iter):
    """Add a dummy 'infos' value to each dict in a data stream."""
    for data_dict in data_iter:
        yield {'infos': {}, **data_dict}
