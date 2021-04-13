import inspect

import pytest
import torch as th

from il_representations import algos
from il_representations.data import read_dataset
from il_representations.envs import auto
from il_representations.test_support.configuration import (
    ENV_CFG_TEST_CONFIGS, ENV_DATA_TEST_CONFIG, REPL_SMOKE_TEST_CONFIG)
from il_representations.test_support.utils import is_representation_learner


@pytest.mark.parametrize("algo", [
    el[1] for el in inspect.getmembers(algos)
    if is_representation_learner(el[1])
])
@pytest.mark.parametrize("env_cfg", ENV_CFG_TEST_CONFIGS)
def test_algo(algo, env_cfg, represent_ex):
    """Check that multiple configurations of `represent_ex` runs to completion and
    the trained representation model serializes correctly.
    """
    bench_available, why = auto.benchmark_is_available(
        env_cfg['benchmark_name'])
    if not bench_available:
        pytest.skip(why)
    run = represent_ex.run(config_updates={
        **REPL_SMOKE_TEST_CONFIG,
        'algo': algo,
        'env_cfg': env_cfg,
        'env_data': ENV_DATA_TEST_CONFIG,
        'debug_return_model': True,
    })
    assert run.status == "COMPLETED"

    with open(run.result["encoder_path"], "rb") as f:
        encoder_clone = th.load(f)

    model: algos.RepresentationLearner = run.result["model"]

    # Get inputs to pass into both the original and deserialized encoders.
    datasets, _ = auto.load_wds_datasets([{"env_cfg": env_cfg}])
    data_loader = read_dataset.datasets_to_loader(
        datasets,
        shuffle=False,
        batch_size=model.batch_size,
        # observations. We can then optionally apply a shuffler that retains a
        nominal_length=2 * model.batch_size,
        # small pool of constructed targets in memory and yields
        max_workers=model.dataset_max_workers,
        # randomly-selected items from that pool (this approximates
        shuffle_buffer_size=model.shuffle_buffer_size,
        preprocessors=(model.target_pair_constructor,),
    )
    batch = next(iter(data_loader))
    # TODO(shwang): I had to copy over some preprocessing calls to get the data
    # into the right format for passing into the model.
    # Maybe it would be worth doing some more refactoring to `_make_data_loader` so that
    # the preprocessing calls are unnecessary.
    obs = model._prep_tensors(batch["context"])
    obs = model._preprocess(obs)
    traj_info = model._prep_tensors(batch["traj_ts_ids"])

    output_orig = model.encoder(obs, traj_info)
    output_clone = encoder_clone(obs, traj_info)

    for x, y in zip(model.encoder.parameters(), encoder_clone.parameters()):
        assert th.allclose(x, y)
    assert th.allclose(output_orig.mean, output_clone.mean)
    assert th.allclose(output_orig.stddev, output_clone.stddev)
