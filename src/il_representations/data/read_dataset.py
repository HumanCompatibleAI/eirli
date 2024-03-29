"""Tools for reading datasets stored in the webdataset format."""

import logging
import os
import pickle
import urllib.parse
import warnings

import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from webdataset.dataset import group_by_keys
from webdataset.gopen import reader
import zstandard


def _zst_open(path):
    """Open zstd-compressed file at `path` using the `zstandard` library."""
    decomp = zstandard.ZstdDecompressor()
    fp = open(path, 'rb')
    reader = decomp.stream_reader(fp, closefd=True, read_across_frames=True)
    return reader


def zst_reader(url, **kw):
    """Custom reader for webdataset that reads zstd-compressed files, or defers
    to built-in webdataset reader for files that are not local or not
    zstd-compressed."""
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme
    if scheme in ("file", ""):
        if scheme == "file":
            path = parsed.path
        else:
            path = url
        lp = path.lower()
        if lp.endswith(".zst") or lp.endswith(".zstd"):
            return _zst_open(path)
    return reader(url, **kw)


class ILRDataset(wds.Dataset):
    """Modification of webdataset's `Dataset` class so that it is suitable for
    il-representations. Specific changes:

    - It looks for a `_metadata.meta.pickle` file at the beginning of the data
      archive & makes it available as a class attribute (`.meta`). This is
      useful for storing dataset-global metadata.
    - (possibly some other stuff, I haven't figured it out yet)
    """
    _meta = None  # this will be set by __init__.meta_pipeline

    def __init__(self, urls, *args, open_fn=zst_reader, initial_pipeline=None,
                 **kwargs):
        def meta_pipeline(data_iter):
            """Pipeline that extracts the first element of the archive to use
            as metadata, then lets the remainder go through the rest of the
            pipeline."""
            first_item = next(data_iter)
            assert isinstance(first_item, tuple) and len(first_item) == 2
            first_item_name, first_item_data = first_item
            if first_item_name != '_metadata.meta.pickle':
                raise ValueError(
                    "data archive does not have '_metadata.meta.pickle' as "
                    "its first file; was it produced with one of the "
                    "mkdataset_* scripts?")
            metadata = pickle.loads(first_item_data)
            self._meta = metadata
            for item in data_iter:
                if item[0] == '_metadata.meta.pickle':
                    # skip metadata files form here on
                    new_meta = pickle.loads(item[1])
                    if new_meta != self._meta:
                        warnings.warn(
                            "Dataset has second metadata object, not equal "
                            "to first: {new_meta!r} != {self._meta!r}")
                    continue
                yield item

        assert len(urls) >= 1, "this requires at least one url"

        if initial_pipeline is None:
            # group_by_keys is part of the default initial pipeline, so we
            # include it here as a default too
            init_pipeline = [meta_pipeline, group_by_keys()]
        else:
            # if initial_pipeline is given to Dataset, then it does not use
            # group_by_keys, so we omit it in the analogous case here
            init_pipeline = [meta_pipeline, initial_pipeline]

        super().__init__(urls, *args, initial_pipeline=init_pipeline,
                         open_fn=open_fn, **kwargs)

    @property
    def meta(self):
        if not self._meta:
            # load dataset just to make sure we have metadata
            next(iter(self))
            assert self._meta is not None, \
                "self._meta should be populated on first sample draw, " \
                "but it was not (may be bug in this class, or empty dataset)"
        return self._meta


class InterleavedDataset(IterableDataset):
    """Randomly interleaves one or more IterableDatasets. Pretends that the
    resulting dataset has length `nominal_length`; underlying datasets are
    looped as needed, like in WDS' ResizeDataset."""
    def __init__(self, datasets, nominal_length, seed=None):
        super().__init__()
        self.rng = np.random.RandomState(seed)
        self.nominal_length = nominal_length
        self.datasets = list(datasets)
        assert len(self.datasets) > 0
        self.iterators = [None] * len(datasets)

    def __iter__(self):
        for _ in range(self.nominal_length):
            chosen = self.rng.choice(len(self.datasets))
            chosen_ds = self.datasets[chosen]
            try:
                # try to get next item
                if self.iterators[chosen] is None:
                    raise StopIteration()
                next_item = next(self.iterators[chosen])
            except StopIteration:
                # refresh iterator if we fail
                self.iterators[chosen] = iter(chosen_ds)
                # if this next() fails then the dataset must be empty (!!)
                next_item = next(self.iterators[chosen])
            yield next_item

    def __len__(self):
        return self.nominal_length


class SubdatasetExtractor:
    def __init__(self, n_trajs=None, n_trans=None):
        assert n_trajs is None or n_trans is None, \
            'Specify one or none of n_traj and n_trans, not both.'
        self.n_trajs = n_trajs
        self.n_trans = n_trans

        if self.n_trajs is not None:
            logging.info(f"Training with {self.n_trajs} trajectories.")
        elif self.n_trans is not None:
            logging.info(f"Training with {self.n_trans} transitions.")
        else:
            logging.info("Training with full data.")

    def __call__(self, data_iter):
        trajectory_ind = 0

        if self.n_trans is not None:
            trans_count = 0
            for step_dict in data_iter:
                trans_count += 1
                yield step_dict

                if trans_count == self.n_trans:
                    break

            assert trans_count == self.n_trans, (self.n_trans, trans_count)
        else:
            for step_dict in data_iter:
                yield step_dict

                if step_dict.get('dones', False):
                    trajectory_ind += 1

                if trajectory_ind == self.n_trajs:
                    break

            assert self.n_trajs is None or trajectory_ind == self.n_trajs, \
                (self.n_trajs, trajectory_ind)


def strip_extensions(dataset):
    """Strip extensions from the keys in dicts generated by ILRDataset."""
    for item in dataset:
        # this might fail if user did not use group_keys() transform on dataset
        # (this is applied by default, but can disappear if you override
        # initial_pipeline)
        assert isinstance(item, dict), type(item)
        new_item = {k.split('.', 1)[0]: v for k, v in item.items()}
        yield new_item


def load_ilr_datasets(file_paths):
    """Load a dataset that has been stored in webdataset format, given a list
    of file paths representing shards of the dataset."""
    # wds doesn't use standard URL parser for some reason. I think they're just
    # looking for a file: prefix and then treating the rest of the string as a
    # file path.
    urls = ['file:' + os.path.abspath(p) for p in file_paths]
    return ILRDataset(urls) \
        .decode() \
        .pipe(strip_extensions)


def datasets_to_loader(datasets, *, batch_size, nominal_length=None,
                       shuffle=True, shuffle_buffer_size=1024, max_workers=1,
                       preprocessors=(), drop_last=True, collate_fn=None):
    """Turn a sequence of webdataset datasets into a single Torch data
    loader that mixes the datasets equally.

    Args:
        datasets ([Dataset]): sequence of datasets to mix.
        batch_size (int): size of batches to yield.
        nominal_length (Optional[int]): supposed length of the dataset (number
            of samples, not number of batches). This governs how long you
            can draw samples from the `DataLoader` for before it raises
            `StopIteration`. If you aren't relying on `DataLoader` raising
            `StopIteration`, then you can make this "length" as large as you
            like.
        shuffle (bool): should we use an intermediate buffer to shuffle
            samples, after applying preprocessors but before forming batches?
        shuffle_buffer_size (int): size of the intermediate buffer to use for
            shuffling.
        preprocessors ([fn]): a list of preprocessors which will be applied to
            each dataset using `.pipe()`. Note that these preprocessors get to
            see samples in the order they were written to disk, which can be
            useful for things like target pair construction.

    Returns
        torch.DataLoader: a Torch DataLoader that returns batches of the
            required size, with elements drawn with equal probability from all
            constituent datasets.
    """

    # For each single-task dataset in the `datasets` list, we first apply a
    # target pair constructor to create targets from the incoming stream of
    # observations. We can then optionally apply a shuffler that retains a
    # small pool of constructed targets in memory and yields
    # randomly-selected items from that pool (this approximates
    # full-dataset shuffling without having to read the whole dataset into
    # memory).

    for sub_ds in datasets:
        for preprocessor in preprocessors:
            # Construct representation learning dataset of correctly paired
            # (context, target) pairs
            sub_ds.pipe(preprocessor)
        if shuffle:
            # TODO(sam): if we're low on memory due to shuffle buffer memory
            # consumption, then consider shuffling *after* interleaving (more
            # complicated, but also takes up less memory).
            sub_ds.shuffle(shuffle_buffer_size)

    if not shuffle:
        assert len(datasets) <= 1, \
            "InterleavedDataset will intrinsically shuffle batches by " \
            "randomly selecting which dataset to draw from at each " \
            "iteration; do not use multi-task training if " \
            f"shuffle_batches=False is required (got {len(datasets)}" \
            "datasets)"
        assert max_workers <= 1, \
            "Using more than one dataset worker may shuffle the " \
            "dataset; got max_workers={max_workers}"
    interleaved_dataset = InterleavedDataset(
        datasets, nominal_length=nominal_length)

    assert not (drop_last and nominal_length < batch_size), \
        f"dropping last batch when nominal_length ({nominal_length}) is " \
        f"smaller than batch size ({batch_size}) will yield an empty dataset"

    dataloader = DataLoader(interleaved_dataset,
                            num_workers=max_workers,
                            batch_size=int(batch_size),
                            drop_last=drop_last,
                            collate_fn=collate_fn)

    return dataloader
