# iterate benchmark over [dm_control, magical]
    # iterate env over ENV_SET[benchmark]
        # iterate algo over algos_to_tune



# algos to tune (and associated hyperparameters)
    # TemporalCPC - representation_dim
    # TemporalCPC + Aug - representation_dim, augmentation type
    # Action Conditioned Temporal CPC - representation_dim, action_encoding_dim, action_embedding_dim
    # IdentityCPC + Aug - Representation dim, augmentation type
    # TemporalCEB - representation_dim, beta, projection_dim, fixed vs learned variance
    # IdentityCEB + Aug - Representation dim, augmentation_type, beta, fixed vs learned variance
    # VAE - representation_dim, beta,
    # Momentum - Augmentation type, momentum weight, batch_negatives [Y/N]


# + projection dim for anything we give decoders