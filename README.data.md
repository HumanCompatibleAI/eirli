# Dataset and environment code README

This document explains the abstractions that we are using to provide a
reasonably uniform internal interface across all of all the benchmarks supported
by the il-representations project (Atari, dm_control, MAGICAL, Minecraft, etc.).

## Sacred ingredients

The data-loading pipeline is configured using three different Sacred
ingredients:

- [`env_cfg_ingredient`](https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/config.py#L10-L68):
  In principle, this ingredient contains all the information necessary to create
  a Gym environment for a specific combination of benchmark and task. The two
  most important config keys are `benchmark_name` (which identifies whether the
  current benchmark is MAGICAL, or dm_control, or something else), and
  `task_name` (which identifies the current task within the selected benchmark;
  e.g. finger-spin or MoveToCorner). There are also some benchmark-specific
  config keys for, e.g., preprocessing.
- [`venv_opts_ingredient`](https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/config.py#L71-L92):
  Additional options required to construct a vecenv (e.g. the number of
  environments to run in parallel).
- [`env_data_ingredient`](https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/config.py#L95-L173):
  Contains paths to data files on disk. Has quite a few dataset-specific keys,
  particularly for loading 'native'-format datasets (as described further down).

Not every script requires every one of the above ingredients, so they have been
separated out to minimise the total number of config variables for each script.

## Creating Gym environments

Gym environments can be created with
[`auto.load_vec_env()`](https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/auto.py#L68-L109).
This uses `env_cfg['benchmark_name']` (from the `env_cfg_ingredient` Sacred
ingredient) to dispatch to a benchmark-specific routine for creating vecenvs.
These benchmark-specific routines make use of both `env_cfg['task_name']` and
(possibly) some benchmark-specific keys in `env_cfg` to, e.g., apply appropriate
preprocessors. In addition, `auto.load_vec_env()` uses `venv_opts` (from the
`venv_opts_ingredient` Sacred ingredient) to determine, e.g., how many
environments the vecenv should run in parallel.

## Loading demonstrations from their 'native' format

Demonstrations for each benchmark were originally generated in a few different
formats. For instance, the MAGICAL reference demonstrations are distributed as
pickles with one file per trajectory, while the Atari demonstrations were saved
as Numpy `.npz` files (for more info, see the [data formats
GDoc](https://docs.google.com/document/d/1YrXFCmCjdK2HK-WFrKNUjx03pwNUfNA6wwkO1QexfwY/edit#heading=h.akt76l1pl1l5).
The
[`auto.load_dataset_dict()`](https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/auto.py#L26-L45)
function provides a uniform interface to these formats.

Like `auto.load_vec_env()`, the `auto.load_dict_dataset()` function uses
`env_cfg['benchmark_name']` to dispatch to a benchmark-specific data-loading
function that is able to read the corresponding on-disk data format. Those
benchmark-specific loading functions in turn look at benchmark-specific config
keys in `env_data` (from `env_data_ingredient`) to locate the demonstrations.
For example, `benchmark_name="magical"` dispatches to
[`envs.magical_envs.load_data()`](https://github.com/HumanCompatibleAI/il-representations/blob/77b557654d1d48a966e84b22d101b06f8ca5b476/src/il_representations/envs/magical_envs.py#L25-L100),
which looks up the current task name (i.e. `env_cfg["task_name"]`) in
`env_data["magical_demo_dirs"]` to determine where the relevant demonstrations
are stored.

Regardless of the value of `env_cfg['benchmark_name']`,
`auto.load_dataset_dict()` always returns a dict with the following keys:

- `obs`: a `T*C*H*W` array of observations associated with states.
- `next_obs`: a `T*C*H*W` array of observations associated with the state
  _after_ the corresponding one in `obs`.
- `acts`: a `T*A_1*A_2*…` array of actions, where `A_1*A_2*…` is the shape of
  the action space.
- `dones`: a length-`T` array of bools indicating whether the corresponding
  state was terminal.

Note that `T` here is the sum of the lengths of all trajectories in the dataset;
trajectories are concatenated together to form each value in the returned
dictionary. It is possible to segment the values back into trajectories by
looking at the `dones` array.

Loading all demonstrations into a single dictionary in memory has one major
advantage, but also a few drawbacks. The advantage is that it's easy to
manipulate the demonstrations: you can figure out how many trajectories you have
by counting the `dones` with Numpy, or randomly index into trajectories when
constructing shuffled batches. However, there are also disadvantages:

- Loading all demonstrations into a dict can use a lot of memory.
- `load_dict_dataset()` relies on the `env_cfg_ingredient` Sacred ingredient,
  which only supports specifying a single training task. Thus it is not easy to
  extend `load_dict_dataset()` so that it can load multitask data (which we
  would like to do for repL!).
- It's hard to invert `load_dict_dataset()` into a function for _saving_
  trajectories, since it needs to support several different (benchmark-specific)
  data formats. However, it would be convenient to have such an inverse
  function, since it would allow us to write benchmark-agnostic code for
  generating new repL training data (e.g. from random rollouts, or saved policies).
  
For these reasons, we also have a second data format…

## The webdataset format

In addition to the in-memory dict format generated by
`auto.load_dict_dataset()`, we also have a second set of independent data-saving
and data-loading machinery based on the
[webdataset](https://github.com/tmbdev/webdataset/) spec/library. The on-disk
format for webdataset (which I'll just call the "webdataset format"), is very
simple: each collection of data is a single tar file, **TODO**
