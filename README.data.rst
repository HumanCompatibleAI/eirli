Dataset and environment README
==============================

*This README current as of 2020-11-29. Some aspects of the data-loading
pipeline will probably change eventually, so expect this doc to be
slightly out of date if you're reading it 2-3 months (or more) in the
future. Bug Sam if there are serious problems with it.*

This document explains the abstractions that we are using to load
demonstration data and create Gym environments. These abstractions are
intended provide a reasonably uniform internal interface across all of
all the benchmarks supported by the il-representations project (Atari,
dm\_control, MAGICAL, Minecraft, etc.).

Sacred ingredients
------------------

The data-loading pipeline is configured using three different Sacred
ingredients:

-  ```env_cfg_ingredient`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/config.py#L10-L68>`__:
   In principle, this ingredient contains all the information necessary
   to create a Gym environment for a specific combination of benchmark
   and task. The two most important config keys are ``benchmark_name``
   (which identifies whether the current benchmark is MAGICAL, or
   dm\_control, or something else), and ``task_name`` (which identifies
   the current task within the selected benchmark; e.g. finger-spin or
   MoveToCorner-Demo-v0). There are also some benchmark-specific config
   keys for, e.g., preprocessing.
-  ```venv_opts_ingredient`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/config.py#L71-L92>`__:
   Additional options required to construct a vecenv (e.g. the number of
   environments to run in parallel).
-  ```env_data_ingredient`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/config.py#L95-L173>`__:
   Contains paths to data files on disk. Has quite a few
   dataset-specific keys, particularly for loading 'native'-format
   datasets (as described further down).

Not every script requires every one of the above ingredients. For
instance, testing a trained policy requires ``env_cfg_ingredient`` to
determine which environment to evaluate on, and ``venv_opts_ingredient``
to construct a vecenv, but not ``env_data_ingredient``. As a result, the
three components have been separated out to minimise the number of
redundant Sacred config options for each script.

Creating Gym environments
-------------------------

Gym environments can be created with
```auto.load_vec_env()`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/auto.py#L68-L109>`__.
This uses ``env_cfg['benchmark_name']`` (from the ``env_cfg_ingredient``
Sacred ingredient) to dispatch to a benchmark-specific routine for
creating vecenvs. The benchmark-specific routines make use of both
``env_cfg['task_name']`` and (possibly) some benchmark-specific keys in
``env_cfg`` to, e.g., apply appropriate preprocessors. In addition,
``auto.load_vec_env()`` uses ``venv_opts`` (from the
``venv_opts_ingredient`` Sacred ingredient) to determine, e.g., how many
environments the vecenv should run in parallel.

Loading demonstrations from their 'native' format
-------------------------------------------------

Demonstrations for each benchmark were originally generated in a few
different formats. For instance, the MAGICAL reference demonstrations
are distributed as pickles, while the Atari demonstrations were saved as
Numpy ``.npz`` files (see the `data formats
GDoc <https://docs.google.com/document/d/1YrXFCmCjdK2HK-WFrKNUjx03pwNUfNA6wwkO1QexfwY/edit#heading=h.akt76l1pl1l5>`__
for more detail). The
```auto.load_dataset_dict()`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/auto.py#L26-L45>`__
function provides a uniform interface to these formats.

Like ``auto.load_vec_env()``, the ``auto.load_dict_dataset()`` function
uses ``env_cfg['benchmark_name']`` to dispatch to a benchmark-specific
data-loading function that is able to read the benchmark's on-disk data
format. Those benchmark-specific loading functions in turn look at
benchmark-specific config keys in ``env_data`` (from
``env_data_ingredient``) to locate the demonstrations. For example,
``benchmark_name="magical"`` dispatches to
```envs.magical_envs.load_data()`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/magical_envs.py#L25-L100>`__,
which looks up the current task name (i.e. ``env_cfg["task_name"]``) in
``env_data["magical_demo_dirs"]`` to determine where the relevant
demonstrations are stored.

Regardless of the value of ``env_cfg['benchmark_name']``,
``auto.load_dataset_dict()`` always returns a dict with the following
keys:

-  ``obs``: an ``N*C*H*W`` array of observations associated with states.
-  ``next_obs``: an ``N*C*H*W`` array of observations associated with
   the state *after* the corresponding one in ``obs``.
-  ``acts``: an ``N*A_1*A_2*…`` array of actions, where ``A_1*A_2*…`` is
   the shape of the action space.
-  ``dones``: an length-\ ``N`` array of bools indicating whether the
   corresponding state was terminal.

Note that ``N`` here is the sum of the lengths of all trajectories in
the dataset; trajectories are concatenated together to form each value
in the returned dictionary. It is possible to segment the values back
into trajectories by looking at the ``dones`` array.

Loading all demonstrations into a single dictionary in memory has one
major advantage, but also a few drawbacks. The advantage is that it's
easy to manipulate the demonstrations: you can figure out how many
trajectories you have with ``np.sum(data_dict["dones"])``, or randomly
index into time steps in order to construct shuffled batches. The three
main disadvantages are:

-  Loading all demonstrations into a dict can use a lot of memory.
-  ``auto.load_dict_dataset()`` relies on the ``env_cfg_ingredient``
   Sacred ingredient, which only supports specifying a single training
   task. Thus it is not easy to extend ``auto.load_dict_dataset()`` so
   that it can load multitask data.
-  It's hard to invert ``auto.load_dict_dataset()`` into a function for
   *saving* trajectories, since it needs to support several different
   (benchmark-specific) data formats. However, it would be convenient to
   have such an inverse function, since that would allow us to write
   benchmark-agnostic code for generating new repL training data (e.g.
   generating training data from random rollouts).

For these reasons, we also have a second data format…

The webdataset format
---------------------

In addition to the in-memory dict format generated by
``auto.load_dict_dataset()``, we also have a second set of independent
data-saving and data-loading machinery based on the
`webdataset <https://github.com/tmbdev/webdataset/>`__ spec/library.
This section briefly explains how webdataset works, and how we use it to
load data for the ``run_rep_learner`` script.

High-level interface and configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Within our codebase, the high-level interface for loading datasets in
the webdataset format is the ```auto.load_wds_datasets()``
function <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/auto.py#L126-L200>`__.
This takes a list of configurations for single-task datasets, and
returns an list containing one webdataset ``Dataset`` for each task. It
is then the responsibility of the calling code to apply any necessary
preprocessing steps to those ``Dataset``\ s (e.g. target pair
construction) and to multiplex the datasets with an
``InterleavedDataset``. These abstractions are explained further down
the page.

The configuration syntax for ``auto.load_wds_datasets()`` is exactly the
syntax used for the ``dataset_configs`` configuration option in
``run_rep_learner.py``, and as such deserves some further explanation.
Each element of the list passed to ``auto.load_wds_datasets()`` is a
dict which may contain the following keys:

.. code:: python

    {
        # the type of data to be loaded
        "type": "demos" | "random" | …,
        # a dictionary containing some subset of configuration keys from `env_cfg_ingredient`
        "env_cfg": {…},
    }

Both the ``"type"`` key and the ``"env_cfg"`` key are optional.
``"type"`` defaults to ``"demos"``, and ``"env_cfg"`` defaults to the
current configuration of ``env_cfg_ingredient``. If any sub-keys are
provided in ``"env_cfg"``, then they are recursively combined with the
current configuration of ``"env_cfg_ingredient"``. This allows one to
define new dataset configurations that override only some aspects of the
current ``"env_cfg_ingredient"`` configuration.

This configuration syntax might be more clear with a few examples:

-  Training on random rollouts and demonstrations using the current
   benchmark name from ``env_cfg_ingredient``:

``python    dataset_configs = [{"type": "demos"}, {"type": "random"}]``
- Training on demos from both the default task from
``env_cfg_ingredient``, and another task called "finger-spin". Notice
that this time the first config dict does not have *any* keys; this is
equivalent to using ``{"type": "demos"}`` as we did above.
``"type": "demos"`` is also implicit in the second dict.

``python    dataset_configs = [{}, {"env_cfg": {"task_name": "finger-spin"}}]``
- Combining the examples above, here is an example that trains on demos
from the current task, random rollouts from the current task, demos from
a second task called ``"finger-spin"``, and random rollouts from a third
task called ``"cheetah-run"``:

``python    dataset_configs = [        {},        {"type": "random"},        {"env_cfg": {"task_name": "finger-spin"}},        {"type": "random", "env_cfg": {"task": "cheetah-run"}},    ]``

Since ``env_cfg_ingredient`` does not allow for specification of data
paths, the configurations passed to ``auto.load_wds_datasets()`` also do
not allow for paths to be overridden. Instead, the data for a given
configuration will always be loaded using the following path template:

::

    <data_root>/processed/<data_type>/<task_key>/<benchmark_name>

``data_root`` is a config variable from ``env_data_ingredient``, and
``data_type`` is the ``"type"`` defined in the dataset config dict.
``"task_key"`` is ``env_cfg["task_name"]`` (which is taken from
``env_cfg_ingredient`` by default, but can be overridden in any of the
config dicts passed to ``auto.load_wds_datasets()``). Likewise,
``benchmark_name`` defaults to ``env_cfg["benchmark_name"]``, but can be
overridden by dataset config dicts.

On-disk format
~~~~~~~~~~~~~~

The webdataset-based on-disk format (which I'll just call the
"webdataset format") is very simple: a dataset is composed of 'shards',
each of which is a single tar archive. Each tar archive contains a list
of files like this:

::

    _metadata.meta.pickle
    frame_000.acts.pickle
    frame_000.dones.pickle
    frame_000.frame.pickle
    frame_000.infos.pickle
    frame_000.next_obs.pickle
    frame_000.obs.pickle
    frame_000.rews.pickle
    frame_001.acts.pickle
    frame_001.dones.pickle
    frame_001.frame.pickle
    frame_001.infos.pickle
    frame_001.next_obs.pickle
    frame_001.obs.pickle
    frame_001.rews.pickle
    frame_002.acts.pickle
    frame_002.dones.pickle
    frame_002.frame.pickle
    frame_002.infos.pickle
    frame_002.next_obs.pickle
    …

For the datasets generated by our code, all shards begin with a
``_metadata.meta.pickle`` file holding metadata identifying a specific
benchmark and task (e.g. it contains the observation space for the task,
as well as a configuration for ``env_data_ingredient`` that can be used
to re-instantiate the whole Gym environment). The remaining files
represent time steps in a combined set of trajectories. For instance,
the ``frame_000.*`` files represent the observation encountered at the
first step of the first trajectory, the action taken, the infos dict
returned, the next observation encountered, etc. As with the arrays
returned by ``auto.load_dict_dataset()``, trajectories are concatenated
together in the tar file, and can be separated back out by inspecting
the ``dones`` values.

*Aside:* users of the webdataset library usually do not include
file-level metadata of the kind stored in ``_metadata.meta.pickle``. Our
code has some additional abstractions (such as
``read_dataset.ILRDataset``) which ensure that the file-level metadata
is accessible from Python, and which also ensure that
``_metadata.meta.pickle`` is not accidentally treated as an additional
"frame" when reading the tar file. This is discussed further below.

Writing datasets in the webdataset format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convenience functions for writing datasets are located in
```data.write_dataset`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/data/write_dataset.py>`__.
In particular, this contains a helper function for extracting metadata
from an ``env_cfg_ingredient`` configuration
(```get_meta_dict()`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/data/write_dataset.py#L21-L49>`__)
and a helper for writing a series of frames to an
appropriately-structured tar archive
(```write_frames()`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/data/write_dataset.py#L52-L71>`__).
These helpers are currently used by two scripts, which are good
resources for understanding how to write webdatasets:

-  ```mkdataset_demos.py`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/scripts/mkdataset_demos.py>`__:
   Converts between dict format and webdataset format. That is, the
   script loads a dataset from its 'native' on-disk format into a dict
   using ``auto.load_dict_dataset()``, then writes the data into a new
   webdataset.
-  ```mkdataset_random.py`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/scripts/mkdataset_random.py>`__:
   Generates random rollouts on a specified environment and then saves
   them into a webdataset.

Loading data: from shard to minibatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main abstraction provided by the webdataset library is the
```Dataset`` <https://github.com/tmbdev/webdataset/blob/b208b15f6a5b14b8e597d5fc182f6945e6390d84/webdataset/dataset.py#L409-L462>`__
class. Given a series of URLs pointing to different shards of a dataset,
this class iterates over the contents over the shards, one URL at a
time. webdataset's ``Dataset`` is a valid subclass of Torch's
``IterableDataset``, so it can be directly passed to Torch's
``DataLoader``. A webdataset ``Dataset`` can also be also be composed
with Python generators in order to create a data preprocessing pipeline.
For repL, our pipeline looks something like this:

1. **Generic decoding/grouping code:** The first stage of the pipeline
   does bookkeeping like decoding ``.pickle`` files in the shard into
   Python objects (instead of yielding raw bytes as training samples!),
   and grouping samples with the same frame prefix (e.g. ``frame000``,
   ``frame001``, etc.). Our code also uses a `special ``Dataset``
   subclass <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/data/read_dataset.py#L13-L71>`__
   that makes the contents of ``_metadata.meta.pickle`` accessible as a
   dataset instance attribute.
2. **Target pair constructor:** After training samples are decoded, they
   can be grouped into context and target pairs for the purpose of repL.
   The ```TargetPairConstructor``
   interface <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/algos/pair_constructors.py#L39-L49>`__
   is simply a generator that processes one sample at a time from the
   dataset iterator. Since samples are written and read in temporal
   order, it is possible for these generators to, e.g., create target
   and context pairs out of temporally adjacent pairs
   (`example <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/algos/pair_constructors.py#L117-L163>`__).
3. **Optional shuffling:** Since webdataset ``Dataset``\ s are
   ``Iterable`` datasets, it is not possible to shuffle the entire
   dataset in-memory. Instead, the repL code can optionally apply a
   pipeline stage that buffers a small, fixed number of samples in
   memory, and pops a randomly-selected sample from this buffer at each
   iteration. This introduces a small degree of randomisation that may
   be helpful for optimisation. Note that this step also breaks temporal
   order, so it must come *after* target pair construction.
4. **Interleaving:** Recall that one of the aims of the
   ``webdataset``-based repL data system was to support multitask
   training. In principle, we could do this by passing shards from
   different datasets to webdataset's ``Dataset`` class. However, since
   shards are iterated over sequentially (modulo the shuffle buffer),
   this would mean that the network would exclusively see samples from
   the first dataset for the first few batches, then exclusively samples
   from the second dataset, and so on. Instead, we create a separate
   webdataset ``Dataset`` for each sub-dataset used for multitask
   training, and then multiplex those ``Dataset``\ s with
   ```InterleavedDataset`` <https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/data/read_dataset.py#L74-L116>`__.
   ``InterleavedDataset`` is an ``IterableDataset`` that repeatedly
   chooses a sub-dataset uniformly at random and yields a single sample
   from that. This ensures that the different sub-datasets are equally
   represented (on average) in each batch.

The steps above yield a single ``IterableDataset`` which can be passed
to Torch's ``DataLoader``. The ``DataLoader`` is then responsible for
combining samples from the iterator into batches, just as it would with
any other ``IterableDataset``.

Adding support for a new benchmark
----------------------------------

These are the rough steps required to add support for a new benchmark:

1. Create benchmark-specific routines for creating vec envs; loading
   data in a dict format; and inferring the equivalent Gym name of an
   environment. Add these to a module in ``il_representations.envs``,
   much like ``il_representations.envs.magical``.
2. Add any required config variables for the new benchmark to
   ``il_representations.envs.config``, and update
   ``il_representations.envs.auto`` so that the routines make use of the
   new config variables to dispatch to the dataset-specific routines in
   ``il_representations.envs.auto``.
3. Update ``il_representations.scripts.il_test`` to do execute
   dataset-specific code is required for evaluation of policies in the
   new environment.
4. Add demonstrations for the new environment to svm and perceptron (in
   ``/scatch/sam/il-demos``). Also update
   ``convert_all_to_new_data_format.sh`` (in
   ``il_representations/scripts/``) to produce webdataset-format
   demonstrations for the new benchmark, and add those to svm/perceptron
   too. Repeat these steps to copy demonstrations to GCP, too. In
   particular, if you copy them to
   ``/scratch/sam/il-representations-gcp-volume/il-demos/`` in svm or
   perceptron then they should get automatically synced to GCP.
5. Finally, add configs for one environment from the new benchmark to
   ``test_support.py``, and add test fixtures to ``tests/data``. This
   will make it possible to unit test the new benchmark. Since these
   data fixtures are stored in the repo, I suggest using only 1-2
   trajectories for each fixture.

