# TODO Should this be in utils instead?
"""Trace & save a network for a MAGICAL environment. Capable of automatically
figuring out where the encoder is."""

import argparse
import json
import os
from typing import Iterator, Optional, Tuple

import torch as th
from torch import nn


# 12 channels, 96x96 image
INPUT_SHAPE = (1, 12, 96, 96)


def load_eval_net(net_path: str) -> nn.Module:
    """Load a network on disk, making sure it ends up on the CPU & in eval
    mode."""
    return th.load(net_path, map_location=th.device('cpu')).cpu().eval()


def trace_encoder(encoder: nn.Module) -> nn.Module:
    """Generate a random example of the appropriate size & use it to trace the
    given network."""
    example_input = th.rand(INPUT_SHAPE)
    return th.jit.trace(encoder, example_input)


NameAndMod = Tuple[str, nn.Module]


def pre_order_children(net: nn.Module) -> Iterator[NameAndMod]:
    """Traverse module tree in pre-order."""
    def _recurse(children: Iterator[NameAndMod]) -> Iterator[NameAndMod]:
        for name, module in children:
            yield name, module
            yield from _recurse(module.named_children())
    yield from _recurse(net.named_children())


def fetch_encoder(net: nn.Module) -> nn.Module:
    # How do I get at the network _inside_ the encoder?
    #
    # - For policies: t.features_extractor.representation_encoder.network
    # - For other stuff: I think you load the encoder directly & use '.network'
    #
    # So, I think walking the children & finding the first thing called
    # '.network' is probably fine.
    for name, module in pre_order_children(net):
        if name == 'network':
            return module
    raise Exception(f"Could not find 'network' descendent in network: {net}")


def get_config_path(module_path: str) -> Optional[str]:
    """Keep walking up the tree until we find a config.json."""
    head = module_path
    while True:
        head, tail = os.path.split(head)
        conf_path = os.path.join(head, 'config.json')
        if os.path.exists(conf_path):
            return conf_path
        if not tail:
            break
    return None  # explicit 'return None' in case of failure


def auto_save_name(module_path: str) -> str:
    """Use config.json files to automatically come up with a name for the given
    encoder."""
    # I can use exp_ident and env name (but not benchmark name I guess)
    config_path = get_config_path(module_path)
    if config_path is None:
        raise Exception(
            f"Could not create an autoname for {module_path} because there "
            "is no config.json in any ancestor directory")
    with open(config_path, 'r') as fp:
        config_dict = json.load(fp)
    seed = str(config_dict['seed'])
    exp_ident = str(config_dict['exp_ident'])
    task_name = str(config_dict['env_cfg']['task_name'])
    # usually above_dir will be either 'repl' or 'il_train'; either way, it
    # gives some clues as to what this run was
    above_dir = os.path.basename(os.path.dirname(os.path.dirname(config_path)))
    above_lookup = {
        'repl': 'repl_enc',
        'il_train': 'policy',
    }
    type_suffix = above_lookup.get(above_dir, 'unk_model')
    return f'{exp_ident}_{task_name}_{type_suffix}_s{seed}.pt'


def main(args: argparse.Namespace) -> None:
    print(f"Loading model from '{args.module_path}'")
    network = load_eval_net(args.module_path)
    encoder = fetch_encoder(network)
    # we trace the encoder with torchscript so that it can be saved
    # and loaded without having `il_representations` installed
    traced_encoder = trace_encoder(encoder)
    save_name = auto_save_name(args.module_path)
    os.makedirs(args.dest_dir, exist_ok=True)
    final_path = os.path.join(args.dest_dir, save_name)
    print(f"Saving loaded model to '{final_path}'")
    traced_encoder.save(final_path)


parser = argparse.ArgumentParser(description=" ".join(__doc__.splitlines()))
parser.add_argument(
    'module_path',
    help='path to encoder or policy to save as a traced network')
parser.add_argument(
    '--dest-dir',
    default='traced_nets/',
    help='destination directory to write things to')

if __name__ == '__main__':
    main(parser.parse_args())
