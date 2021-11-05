from il_representations.algos import (ActionPrediction, DynamicsPrediction,
                                      InverseDynamicsPrediction, SimCLR,
                                      TemporalCPC, VariationalAutoencoder)


def make_jt_configs(train_ex):
    @train_ex.named_config
    def repl_noid():
        # Adds a useless ID objective that won't influence training
        # (hence 'noid' rather than 'id'). Used as a control, since I can't
        # disable repL entirely yet.
        repl = {
            'algo': InverseDynamicsPrediction,
            'algo_params': {
                'batch_size': 2,
            },
        }
        repl_weight = 0.0

        _ = locals()
        del _

    @train_ex.named_config
    def repl_noid_noaugs():
        # like repl_noid, but disables all augmentations too
        repl = {
            'algo': InverseDynamicsPrediction,
            'algo_params': {
                'batch_size': 2,
                'augmenter_kwargs': {
                    'augmenter_spec': None,
                },
            },
        }
        repl_weight = 0.0
        bc = {'augs': None}

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
                'batch_size': 64,
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
                'batch_size': 64,
            },
        }
        repl_weight = 1.0

        _ = locals()
        del _

    @train_ex.named_config
    def repl_simclr():
        # simclr
        repl = {
            'algo': SimCLR,
            'algo_params': {
                'batch_size': 384,
            },
        }
        repl_weight = 1.0
        _ = locals()
        del _

    @train_ex.named_config
    def repl_simclr_192():
        # simclr, 192 batch size
        repl = {
            'algo': SimCLR,
            'algo_params': {
                'batch_size': 192,
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
            'algo_params': {
                'batch_size': 64,
            },
        }
        repl_weight = 1.0

        _ = locals()
        del _

    @train_ex.named_config
    def repl_ap():
        # action prediction (aka BC)
        repl = {
            'algo': ActionPrediction,
            'algo_params': {
                'batch_size': 64,
            },
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
                'batch_size': 384,
            },
        }
        repl_weight = 1.0
        _ = locals()
        del _

    @train_ex.named_config
    def repl_tcpc8_192():
        # temporal cpc (8 steps, batch size 192)
        repl = {
            'algo': TemporalCPC,
            'algo_params': {
                'target_pair_constructor_kwargs': {
                    'temporal_offset': 8,
                },
                'batch_size': 192,
            },
        }
        repl_weight = 1.0
        _ = locals()
        del _

    @train_ex.named_config
    def augs_neurips_repl_bc():
        """Default augmentations used for NeurIPS benchmarks track."""
        repl = {
            'algo_params': {
                'augmenter_kwargs': {
                    'augmenter_spec':
                    'translate,rotate,gaussian_blur,color_jitter_mid'
                },
            },
        }
        bc = {'augs': 'translate,rotate,gaussian_blur,color_jitter_mid'}
        _ = locals()
        del _

    @train_ex.named_config
    def env_use_magical():
        env_cfg = {
            'benchmark_name': 'magical',
            'task_name': 'CHANGE_TASK_NAME',
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

    @train_ex.named_config
    def repl_data_demos():
        repl = {
            'dataset_configs': [
                {'type': 'demos'},
            ]
        }

        _ = locals()
        del _

    @train_ex.named_config
    def repl_data_demos_random():
        repl = {
            'dataset_configs': [
                {'type': 'demos'},
                {'type': 'random'},
            ]
        }

        _ = locals()
        del _

    @train_ex.named_config
    def repl_data_5demos_random():
        """Use 5 demos + all available random rollout data for repL."""
        repl = {
            'dataset_configs': [
                {'type': 'demos', 'env_data': {'wds_n_trajs': 5}},
                {'type': 'random'},
            ]
        }

        _ = locals()
        del _

    @train_ex.named_config
    def disable_extra_saves_and_eval():
        """Disable regular batch saves, model saves, and evaluation. Model
        saving and evaluation will only run at the end."""
        model_save_interval = None
        bc = {
            'short_eval_interval': None,
        }
        repl = {
            'batch_save_interval': None,
        }
        locals()

    @train_ex.named_config
    def bc_data_5demos():
        """Use 5 demos for IL."""
        bc = {'dataset_configs': [
            {'type': 'demos', 'env_data': {'wds_n_trajs': 5}},
        ]}

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
