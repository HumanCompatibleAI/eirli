import ray

from il_representations.test_support.configuration import CHAIN_CONFIG


def test_chain(chain_ex, file_observer):
    try:
        chain_ex.run(config_updates=CHAIN_CONFIG)
    finally:
        # always shut down Ray, in case we get a test failure
        if ray.is_initialized():
            ray.shutdown()
