from il_representations.algos import TemporalOffsetPairConstructor
from il_representations.algos.encoders import ActionEncodingEncoder

# TODO make it possible to modify algorithms essential components albeit with warning


# Assume changes such that momentum=True, projection=True are true for all algorithms

# 1 - TemporalCPC - Random
conf1 = {'algo': 'TemporalCPC',
         'use_random_rollouts': True}

# 2 - Temporal CPC

conf2 = {'algo': 'TemporalCPC',
         'use_random_rollouts': False}

# 3 - Augmentation CPC

# Is this just MoCO?

# 4 - Temporal AutoEncoder
conf4 = {'algo': 'VariationalAutoencoder',
        'algo_params': {
            'target_pair_constructor': TemporalOffsetPairConstructor, # make this actually parse strings correctly,
            'loss_kwargs': {'beta': 0},
        'use_random_rollouts': False
        }}

# 5 - Autoencoder

conf5 = {'algo': 'VariationalAutoencoder',
        'algo_params': {
            'loss_kwargs': {'beta': 0}
        },
        'use_random_rollouts': False}

# 6 - VAE

conf6 = {'algo': 'VariationalAutoencoder', 'use_random_rollouts': False}

# 7 - TemporalCEB - Beta = 0.01
conf7 = {'algo': 'FixedVarianceCEB',
        'algo_params': {
            'loss_kwargs': {'beta': 0.01}
        },
         'use_random_rollouts': False}

# 8 - TemporalCEB - Beta = 1

conf8 = {'algo': 'FixedVarianceCEB', # Currently CEB is temporal by default
        'algo_params': {
            'loss_kwargs': {'beta': 1.0}
        },
         'use_random_rollouts': False}

# 9 - TemporalVAE - Beta = 0.01
conf9 = {'algo': 'VariationalAutoencoder',
        'algo_params': {
            'target_pair_constructor': TemporalOffsetPairConstructor, # TODO make this actually parse strings correctly,
            'loss_kwargs': {'beta': 0.01}
        },
         'use_random_rollouts': False}

# 10 - TemporalVAE - Beta = 1
conf10 = {'algo': 'VariationalAutoencoder',
        'algo_params': {
            'target_pair_constructor': TemporalOffsetPairConstructor, # TODO make this actually parse strings correctly,
            'loss_kwargs': {'beta': 1.0}
        },
          'use_random_rollouts': False}

# 11 - TemporalCPC - (t+2)
conf11 = {'algo': 'TemporalCPC',
         'use_random_rollouts': False,
         'algo_params': {
             'target_pair_constructor_kwargs': {
                 'temporal_offset': 2
             }
         }}

# 12 - TemporalCPC - (t+5)
conf12 = {'algo': 'TemporalCPC',
         'use_random_rollouts': False,
         'algo_params': {
             'target_pair_constructor_kwargs': {
                 'temporal_offset': 5
             }
         }}

# 13 - Action Conditioned TemporalCPC - (t+1)
conf13 = {'algo': 'ActionConditionedTemporalCPC',
         'use_random_rollouts': False
          }

# 14 - Action Conditioned TemporalCPC - (t+2)
conf14 = {'algo': 'ActionConditionedTemporalCPC',
         'use_random_rollouts': False,
          'algo_params': {
              'target_pair_constructor_kwargs': {
                  'temporal_offset': 2
              }
          }}

# 15 - Action Conditioned TemporalCPC - (t+5)
conf15 = {'algo': 'ActionConditionedTemporalCPC',
         'use_random_rollouts': False,
          'algo_params': {
              'target_pair_constructor_kwargs': {
                  'temporal_offset': 5
              }
          }}
# 16 - Action Conditioned TemporalVAE - (t+1)

conf16 = {
    'algo': 'ActionConditionedTemporalVAE',
    'use_random_rollouts': False,

}