from il_representations.algos import TemporalOffsetPairConstructor
from il_representations.algos.encoders import ActionEncodingEncoder
from il_representations.scripts.run_rep_learner import represent_ex

# TODO make it possible to modify algorithms essential components albeit with warning


# Assume changes such that momentum=True, projection=True are true for all algorithms
@represent_ex.named_config
def condition_one_temporal_cpc_random():
    # Temporal CPC with random demonstrations
    algo = 'TemporalCPC'
    use_random_rollouts = True
    _ = locals()
    del _


@represent_ex.named_config
def condition_two_temporal_cpc():
    algo = 'TemporalCPC'
    use_random_rollouts = False
    _ = locals()
    del _


@represent_ex.named_config
def condition_three_augmentation_cpc():
    # TODO figure out how this is different from MoCo
    algo = 'SimCLR'
    use_random_rollouts = False
    _ = locals()
    del _

# Is this just MoCO?

@represent_ex.named_config
def condition_four_temporal_autoencoder():
    algo = 'VariationalAutoencoder'
    use_random_rollouts = False
    algo_params = {
            'target_pair_constructor': TemporalOffsetPairConstructor,
            'loss_kwargs': {'beta': 0}}
    _ = locals()
    del _



@represent_ex.named_config
def condition_five_autoencoder():
    algo = 'VariationalAutoencoder'
    use_random_rollouts = False
    algo_params = {
            'loss_kwargs': {'beta': 0},
                    }
    _ = locals()
    del _


@represent_ex.named_config
def condition_six_vae():
    algo = 'VariationalAutoencoder'
    use_random_rollouts = False
    algo_params = {
        'loss_kwargs': {'beta': 0.01}} # TODO What is a good default beta here?
    _ = locals()
    del _


@represent_ex.named_config
def condition_seven_temporal_ceb_lowbeta():
    algo = 'FixedVarianceCEB'
    algo_params = {'loss_kwargs': {'beta': 0.01}}
    use_random_rollouts = False
    _ = locals()
    del _

@represent_ex.named_config
def condition_eight_temporal_ceb_highbeta():
    algo = 'FixedVarianceCEB'
    algo_params = {'loss_kwargs': {'beta': 1.0}}
    use_random_rollouts = False
    _ = locals()
    del _



@represent_ex.named_config
def condition_nine_temporal_vae_lowbeta():
    algo = 'VariationalAutoencoder'
    algo_params = {'target_pair_constructor': TemporalOffsetPairConstructor, # TODO make this actually parse strings correctly,
                    'loss_kwargs': {'beta': 0.01}}
    use_random_rollouts = False
    _ = locals()
    del _

@represent_ex.named_config
def condition_ten_temporal_vae_highbeta():
    algo = 'VariationalAutoencoder'
    algo_params = {'target_pair_constructor': TemporalOffsetPairConstructor, # TODO make this actually parse strings correctly,
                    'loss_kwargs': {'beta': 1.0}}
    use_random_rollouts = False
    _ = locals()
    del _


@represent_ex.named_config
def condition_eleven_temporal_cpc_2():
    algo = 'TemporalCPC'
    use_random_rollouts = False
    algo_params = {'target_pair_constructor_kwargs': {'temporal_offset': 2}}
    _ = locals()
    del _


@represent_ex.named_config
def condition_twelve_temporal_cpc_5():
    algo = 'TemporalCPC'
    use_random_rollouts = False
    algo_params = {'target_pair_constructor_kwargs': {'temporal_offset': 5}}
    _ = locals()
    del _


@represent_ex.named_config
def condition_thirteen_ac_temporal_cpc():
    algo = 'ActionConditionedTemporalCPC'
    use_random_rollouts = False
    _ = locals()
    del _


@represent_ex.named_config
def condition_fourteen_ac_temporal_cpc_2():
    algo = 'ActionConditionedTemporalCPC'
    use_random_rollouts = False
    algo_params = {'target_pair_constructor_kwargs': {'temporal_offset': 2}}
    _ = locals()
    del _


@represent_ex.named_config
def condition_fifteen_ac_temporal_cpc_5():
    algo = 'ActionConditionedTemporalCPC'
    use_random_rollouts = False
    algo_params = {'target_pair_constructor_kwargs': {'temporal_offset': 5}}
    _ = locals()
    del _

@represent_ex.named_config
def condition_sixteen_ac_temporal_vae():
    algo = 'ActionConditionedTemporalVAE'
    use_random_rollouts = False
    _ = locals()
    del _
