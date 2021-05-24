from il_representations.algos import (DynamicsPrediction,
                                      InverseDynamicsPrediction,
                                      VariationalAutoencoder,
                                      TemporalCPC)


def make_jt_configs(train_ex):
    @train_ex.named_config
    def repl_noid():
        # Adds a useless ID objective that won't influence training
        # (hence 'noid' rather than 'id'). Used as a control, since I can't
        # disable repL entirely yet.
        repl = {
            'algo': InverseDynamicsPrediction,
            'algo_params': {
                'batch_size': 2
            },
        }
        repl_weight = 0.0

        _ = locals()
        del _

    @train_ex.named_config
    def repl_vae():
        # VAE
        repl = {
            'algo': VariationalAutoencoder,
            'algo_params': {
                'decoder_kwargs': {
                    'encoder_arch_key': 'MAGICALCNN',
                },
            },
        }
        repl_weight = 1.0

        _ = locals()
        del _

    @train_ex.named_config
    def repl_fd():
        # forward dynamics
        repl = {
            'algo': DynamicsPrediction,
            'algo_params': {
                'decoder_kwargs': {
                    'encoder_arch_key': 'MAGICALCNN',
                },
            },
        }
        repl_weight = 1.0

        _ = locals()
        del _

    @train_ex.named_config
    def repl_id():
        # inverse dynamics
        repl = {
            'algo': InverseDynamicsPrediction,
        }
        repl_weight = 1.0

        _ = locals()
        del _

    @train_ex.named_config
    def repl_tcpc8():
        # temporal cpc (8 steps)
        repl = {
            'algo': TemporalCPC,
            'algo_params': {
                'target_pair_constructor_kwargs': {
                    'temporal_offset': 8,
                },
            },
        }
        repl_weight = 1.0

        _ = locals()
        del _

    @train_ex.named_config
    def env_use_magical():
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'MatchRegions-Demo-v0',
            'magical_remove_null_actions': False,
        }

        _ = locals()
        del _

    @train_ex.named_config
    def env_use_dm_control():
        env_cfg = {
            'benchmark_name': 'dm_control',
            'task_name': 'reacher-easy',
        }

        _ = locals()
        del _

    def inner_scope():
        for task_name in [
                'MoveToCorner-Demo-v0', 'MoveToRegion-Demo-v0',
                'MatchRegions-Demo-v0', 'ClusterColour-Demo-v0',
                'ClusterShape-Demo-v0', 'MakeLine-Demo-v0',
                'FixColour-Demo-v0', 'FindDupe-Demo-v0',
        ]:
            prefix = task_name.split('-')[0]
            prefix_lower = prefix.lower()

            # this namedconfig is to train on demos from both demo variant and
            # test variant
            train_ex.add_named_config(
                f'repl_data_{prefix_lower}_demos_test', {
                    'repl': {
                        'dataset_configs':
                        [{
                            'type': 'demos',
                            'env_cfg': {
                                'benchmark_name': 'magical',
                                'task_name': tn,
                            },
                        } for tn in
                         [f'{prefix}-Demo-v0', f'{prefix}-TestAll-v0']]
                    }
                })

            # similar, but for BC rather than repL
            train_ex.add_named_config(
                f'bc_data_{prefix_lower}_demos_test', {
                    'bc': {
                        'dataset_configs':
                        [{
                            'type': 'demos',
                            'env_cfg': {
                                'benchmark_name': 'magical',
                                'task_name': tn,
                            },
                        } for tn in
                         [f'{prefix}-Demo-v0', f'{prefix}-TestAll-v0']]
                    }
                })

    # put this in another scope for hygiene
    inner_scope()
