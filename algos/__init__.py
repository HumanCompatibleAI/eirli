from .representation_learner import RepresentationLearner
from .encoders import CNNEncoder, MomentumEncoder, InverseDynamicsEncoder, DynamicsEncoder, RecurrentEncoder
from .decoders import ProjectionHead, NoOp, MomentumProjectionHead, BYOLProjectionHead, ActionConditionedVectorDecoder
from .losses import SymmetricContrastiveLoss, AsymmetricContrastiveLoss, MSELoss
from .augmenters import AugmentContextAndTarget, AugmentContextOnly
from .pair_constructors import IdentityPairConstructor, TemporalOffsetPairConstructor
from .batch_extenders import QueueBatchExtender
from .optimizers import LARS


class SimCLR(RepresentationLearner):
    """
    Implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    https://arxiv.org/abs/2002.05709

    This method works by using a contrastive loss to push together representations of two differently-augmented
    versions of the same image. In particular, it uses a symmetric contrastive loss, which compares the
    (target, context) similarity against similarity of context with all other targets, and also similarity
     of target with all other contexts.
    """
    def __init__(self, env, log_dir, **kwargs):
        super(SimCLR, self).__init__(env=env,
                                     log_dir=log_dir,
                                     encoder=CNNEncoder,
                                     decoder=ProjectionHead,
                                     loss_calculator=SymmetricContrastiveLoss,
                                     augmenter=AugmentContextAndTarget,
                                     target_pair_constructor=IdentityPairConstructor,
                                     **kwargs)


class TemporalCPC(RepresentationLearner):
    def __init__(self, env, log_dir, temporal_offset=1, **kwargs):
        """
        Implementation of a non-recurrent version of CPC: Contrastive Predictive Coding
        https://arxiv.org/abs/1807.03748

        By default, augments only the context, but can be modified to augment both context and target.
        """
        target_pair_constructor_kwargs = kwargs.get('target_pair_constructor_kwargs', {})
        assert 'temporal_offset' not in target_pair_constructor_kwargs, "To control temporal_offset, pass parameter directly into constructor"
        target_pair_constructor_kwargs.update({'temporal_offset': temporal_offset})
        super(TemporalCPC, self).__init__(env=env,
                                          log_dir=log_dir,
                                          encoder=CNNEncoder,
                                          decoder=NoOp,
                                          loss_calculator=AsymmetricContrastiveLoss,
                                          target_pair_constructor=TemporalOffsetPairConstructor,
                                          target_pair_constructor_kwargs=target_pair_constructor_kwargs,
                                          **kwargs)


class RecurrentCPC(RepresentationLearner):
    """
    Implementation of a recurrent version of CPC: Contrastive Predictive Coding
    https://arxiv.org/abs/1807.03748

    The encoder first encodes individual frames for both context and target, and then, for the context,
    builds up a recurrent representation of all prior frames in the same trajectory, to use to predict the target.

    By default, augments only the context, but can be modified to augment both context and target.
    """
    def __init__(self, env, log_dir, **kwargs):
        super(RecurrentCPC, self).__init__(env=env,
                                           log_dir=log_dir,
                                           encoder=RecurrentEncoder,
                                           decoder=NoOp,
                                           loss_calculator=AsymmetricContrastiveLoss,
                                           target_pair_constructor=IdentityPairConstructor,
                                           shuffle_batches=False,
                                           **kwargs)


class MoCo(RepresentationLearner):
    """
    Implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, env, log_dir, queue_size=8192, **kwargs):
        batch_extender_kwargs = kwargs.get('batch_extender_kwargs', {})
        assert 'queue_size' not in batch_extender_kwargs, "To control queue_size, pass parameter directly into constructor"
        batch_extender_kwargs.update({'queue_size': queue_size})
        super(MoCo, self).__init__(env=env,
                                   log_dir=log_dir,
                                   encoder=MomentumEncoder,
                                   decoder=NoOp,
                                   loss_calculator=AsymmetricContrastiveLoss,
                                   augmenter=AugmentContextAndTarget,
                                   target_pair_constructor=TemporalOffsetPairConstructor,
                                   batch_extender=QueueBatchExtender,
                                   batch_extender_kwargs=batch_extender_kwargs,
                                   **kwargs)


class MoCoWithProjection(RepresentationLearner):
    """
    Implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
    https://arxiv.org/abs/1911.05722

    Includes an additional projection head atop the representation and before the prediction
    """

    def __init__(self, env, log_dir, queue_size=8192, **kwargs):
        batch_extender_kwargs = kwargs.get('batch_extender_kwargs', {})
        assert 'queue_size' not in batch_extender_kwargs, "To control queue_size, pass parameter directly into constructor"
        batch_extender_kwargs.update({'queue_size': queue_size})
        super(MoCoWithProjection, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=MomentumEncoder,
                                                 decoder=MomentumProjectionHead,
                                                 loss_calculator=AsymmetricContrastiveLoss,
                                                 augmenter=AugmentContextAndTarget,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 batch_extender=QueueBatchExtender,
                                                 batch_extender_kwargs=batch_extender_kwargs,
                                                 **kwargs)


class DynamicsPrediction(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        target_pair_constructor_kwargs = kwargs.get('target_pair_constructor_kwargs', {})
        if 'mode' in target_pair_constructor_kwargs:
            raise Warning(
                f"target_pair_constructor `mode` param must be set to `dynamics`, overwriting current value {target_pair_constructor_kwargs.get('mode')}")
        target_pair_constructor_kwargs.update({'mode': 'dynamics'})
        super(DynamicsPrediction, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=DynamicsEncoder,
                                                 decoder=NoOp, # Should be a pixel decoder that takes in action, currently errors
                                                 loss_calculator=MSELoss,
                                                 augmenter=AugmentContextOnly,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 target_pair_constructor_kwargs=target_pair_constructor_kwargs,
                                                 **kwargs)


class InverseDynamicsPrediction(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        target_pair_constructor_kwargs = kwargs.get('target_pair_constructor_kwargs', {})
        if 'mode' in target_pair_constructor_kwargs:
            raise Warning(
                f"target_pair_constructor `mode` param must be set to `inverse_dynamics`, overwriting current value {target_pair_constructor_kwargs.get('mode')}")
        target_pair_constructor_kwargs.update({'mode': 'inverse_dynamics'})
        super(InverseDynamicsPrediction, self).__init__(env=env,
                                                        log_dir=log_dir,
                                                        encoder=InverseDynamicsEncoder,
                                                        decoder=NoOp, # Should be a action decoder that takes in next obs representation
                                                        loss_calculator=MSELoss,
                                                        augmenter=AugmentContextOnly,
                                                        target_pair_constructor=TemporalOffsetPairConstructor,
                                                        target_pair_constructor_kwargs=target_pair_constructor_kwargs,
                                                        **kwargs)


class BYOL(RepresentationLearner):
    """Implementation of Bootstrap Your Own Latent: https://arxiv.org/pdf/2006.07733.pdf

    The hyperparameters do _not_ match those in the paper (see Section 3.2).
    Most notably, the momentum weight is set to a constant, instead of annealed
    from 0.996 to 1 via cosine scheduling.
    """
    def __init__(self, env, log_dir, **kwargs):
        super(BYOL, self).__init__(env=env,
                                   log_dir=log_dir,
                                   encoder=MomentumEncoder,
                                   decoder=BYOLProjectionHead,
                                   loss_calculator=MSELoss,
                                   augmenter=AugmentContextAndTarget,
                                   target_pair_constructor=IdentityPairConstructor,
                                   **kwargs)


class ActionConditionedTemporalCPC(RepresentationLearner):
    """
    Implementation of reinforcement-learning-specific variant of Temporal CPC which adds a projection layer on top
    of the learned representation which integrates an encoding of the actions taken between time (t) and whatever
    time (t+k) is specified in temporal_offset and used for pulling out the target frame. This, notionally, allows
    the algorithm to construct frame representations that are action-independent, rather than marginalizing over an
    expected policy, as might need to happen if the algorithm needed to predict the frame at time (t+k) over any
    possible action distribution.
    """
    def __init__(self, env, log_dir, temporal_offset=1, shuffle_batches=False, **kwargs):
        target_pair_constructor_kwargs = kwargs.get('target_pair_constructor_kwargs', {})
        decoder_kwargs = kwargs.get('decoder_kwargs', {})
        assert 'temporal_offset' not in target_pair_constructor_kwargs, "To control temporal_offset, pass parameter directly into constructor"

        assert 'action_space' not in decoder_kwargs, "Decoder action_space must be governed by env"
        if 'mode' in target_pair_constructor_kwargs:
            raise Warning(
                f"target_pair_constructor `mode` param must be set to `inverse_dynamics`, overwriting current value {target_pair_constructor_kwargs.get('mode')}")
        target_pair_constructor_kwargs.update({"temporal_offset": temporal_offset, "mode": "dynamics"})
        decoder_kwargs.update({'action_space': env.action_space})

        super(ActionConditionedTemporalCPC, self).__init__(env=env,
                                                           log_dir=log_dir,
                                                           target_pair_constructor=TemporalOffsetPairConstructor,
                                                           target_pair_constructor_kwargs=target_pair_constructor_kwargs,
                                                           encoder=CNNEncoder,
                                                           decoder=ActionConditionedVectorDecoder,
                                                           decoder_kwargs=decoder_kwargs,
                                                           loss_calculator=AsymmetricContrastiveLoss,
                                                           shuffle_batches=shuffle_batches,
                                                           **kwargs)

## Algos that should not be run in all-algo test because they are not yet finished
WIP_ALGOS = [DynamicsPrediction, InverseDynamicsPrediction]
