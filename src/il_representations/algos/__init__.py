from il_representations.algos.representation_learner import RepresentationLearner
from il_representations.algos.encoders import MomentumEncoder, InverseDynamicsEncoder, DynamicsEncoder, RecurrentEncoder, StochasticEncoder, DeterministicEncoder
from il_representations.algos.decoders import ProjectionHead, NoOp, MomentumProjectionHead, BYOLProjectionHead, ActionConditionedVectorDecoder, TargetProjection
from il_representations.algos.losses import SymmetricContrastiveLoss, AsymmetricContrastiveLoss, MSELoss, CEBLoss, \
    QueueAsymmetricContrastiveLoss, BatchAsymmetricContrastiveLoss

from il_representations.algos.augmenters import AugmentContextAndTarget, AugmentContextOnly, NoAugmentation
from il_representations.algos.pair_constructors import IdentityPairConstructor, TemporalOffsetPairConstructor
from il_representations.algos.batch_extenders import QueueBatchExtender
from il_representations.algos.optimizers import LARS
import inspect

DEFAULT_HARDCODED_PARAMS = ['encoder', 'decoder', 'loss_calculator', 'augmenter', 'target_pair_constructor']


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }



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
        self.clean_kwargs(kwargs)

        super().__init__(env=env,
                                     log_dir=log_dir,
                                     encoder=DeterministicEncoder,
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
        self.clean_kwargs(kwargs)
        self.update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                          {'temporal_offset': temporal_offset})
        super().__init__(env=env,
                                          log_dir=log_dir,
                                          encoder=DeterministicEncoder,
                                          decoder=NoOp,
                                          loss_calculator=BatchAsymmetricContrastiveLoss,
                                          target_pair_constructor=TemporalOffsetPairConstructor,
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
        self.clean_kwargs(kwargs)
        kwargs['shuffle_batches'] = False
        super().__init__(env=env,
                                           log_dir=log_dir,
                                           encoder=RecurrentEncoder,
                                           decoder=NoOp,
                                           loss_calculator=BatchAsymmetricContrastiveLoss,
                                           target_pair_constructor=TemporalOffsetPairConstructor,
                                           **kwargs)


class MoCo(RepresentationLearner):
    """
    Implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, env, log_dir, queue_size=8192, **kwargs):
        hardcoded_params = DEFAULT_HARDCODED_PARAMS + ['batch_extender']
        self.clean_kwargs(kwargs, hardcoded_params)
        self.update_kwarg_dict(kwargs, 'batch_extender_kwargs', {'queue_size': queue_size})
        super(MoCo, self).__init__(env=env,
                                   log_dir=log_dir,
                                   encoder=MomentumEncoder,
                                   decoder=NoOp,
                                   loss_calculator=QueueAsymmetricContrastiveLoss,
                                   augmenter=AugmentContextAndTarget,
                                   target_pair_constructor=TemporalOffsetPairConstructor,
                                   batch_extender=QueueBatchExtender,
                                   **kwargs)


class MoCoWithProjection(RepresentationLearner):
    """
    Implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
    https://arxiv.org/abs/1911.05722

    Includes an additional projection head atop the representation and before the prediction
    """

    def __init__(self, env, log_dir, queue_size=8192, **kwargs):
        hardcoded_params = DEFAULT_HARDCODED_PARAMS + ['batch_extender']
        self.clean_kwargs(kwargs, hardcoded_params)

        self.update_kwarg_dict(kwargs, 'batch_extender_kwargs', {'queue_size': queue_size})
        super(MoCoWithProjection, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=MomentumEncoder,
                                                 decoder=MomentumProjectionHead,
                                                 loss_calculator=QueueAsymmetricContrastiveLoss,
                                                 augmenter=AugmentContextAndTarget,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 batch_extender=QueueBatchExtender,
                                                 **kwargs)


class DynamicsPrediction(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        self.clean_kwargs(kwargs)

        self.update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                                                           {'mode': 'dynamics'})
        super(DynamicsPrediction, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=DynamicsEncoder,
                                                 decoder=NoOp, # Should be a pixel decoder that takes in action, currently errors
                                                 loss_calculator=MSELoss,
                                                 augmenter=AugmentContextOnly,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 **kwargs)

    def learn(self, dataset, training_epochs):
        raise NotImplementedError("DynamicsPrediction is not yet implemented")


class InverseDynamicsPrediction(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        self.clean_kwargs(kwargs)
        self.update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                                                           {'mode': 'inverse_dynamics'})
        super(InverseDynamicsPrediction, self).__init__(env=env,
                                                        log_dir=log_dir,
                                                        encoder=InverseDynamicsEncoder,
                                                        decoder=NoOp, # Should be a action decoder that takes in next obs representation
                                                        loss_calculator=MSELoss,
                                                        augmenter=AugmentContextOnly,
                                                        target_pair_constructor=TemporalOffsetPairConstructor,
                                                        **kwargs)

    def learn(self, dataset, training_epochs):
        raise NotImplementedError("InverseDynamicsPrediction is not yet implemented")


class BYOL(RepresentationLearner):
    """Implementation of Bootstrap Your Own Latent: https://arxiv.org/pdf/2006.07733.pdf

    The hyperparameters do _not_ match those in the paper (see Section 3.2).
    Most notably, the momentum weight is set to a constant, instead of annealed
    from 0.996 to 1 via cosine scheduling.
    """
    def __init__(self, env, log_dir, **kwargs):
        self.clean_kwargs(kwargs)
        super(BYOL, self).__init__(env=env,
                                   log_dir=log_dir,
                                   encoder=MomentumEncoder,
                                   decoder=BYOLProjectionHead,
                                   loss_calculator=MSELoss,
                                   augmenter=AugmentContextAndTarget,
                                   target_pair_constructor=IdentityPairConstructor,
                                   **kwargs)


class CEB(RepresentationLearner):
    """
    """
    def __init__(self, env, log_dir, **kwargs):
        self.clean_kwargs(kwargs)
        super(CEB, self).__init__(env=env,
                                  log_dir=log_dir,
                                  encoder=StochasticEncoder,
                                  decoder=NoOp,
                                  loss_calculator=CEBLoss,
                                  augmenter=NoAugmentation,
                                  target_pair_constructor=TemporalOffsetPairConstructor,
                                  **kwargs)

class FixedVarianceCEB(RepresentationLearner):
    """
    """
    def __init__(self, env, log_dir, **kwargs):
        self.clean_kwargs(kwargs)
        super(FixedVarianceCEB, self).__init__(env=env,
                                  log_dir=log_dir,
                                  encoder=DeterministicEncoder,
                                  decoder=NoOp,
                                  loss_calculator=CEBLoss,
                                  augmenter=AugmentContextAndTarget,
                                  target_pair_constructor=TemporalOffsetPairConstructor,
                                  **kwargs)

class FixedVarianceTargetProjectedCEB(RepresentationLearner):
    """
    """
    def __init__(self, env, log_dir, **kwargs):
        self.clean_kwargs(kwargs)
        super(FixedVarianceTargetProjectedCEB, self).__init__(env=env,
                                                              log_dir=log_dir,
                                                              encoder=DeterministicEncoder,
                                                              decoder=TargetProjection,
                                                              loss_calculator=CEBLoss,
                                                              augmenter=AugmentContextAndTarget,
                                                              target_pair_constructor=TemporalOffsetPairConstructor,
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
        self.clean_kwargs(kwargs, self.__class__)
        kwargs['preprocess_extra_context'] = False
        self.update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                                                           {"temporal_offset": temporal_offset, "mode": "dynamics"})

        self.update_kwarg_dict(kwargs, 'decoder_kwargs',
                                           {'action_space': env.action_space})

        super(ActionConditionedTemporalCPC, self).__init__(env=env,
                                                           log_dir=log_dir,
                                                           target_pair_constructor=TemporalOffsetPairConstructor,
                                                           encoder=DeterministicEncoder,
                                                           decoder=ActionConditionedVectorDecoder,
                                                           loss_calculator=BatchAsymmetricContrastiveLoss,
                                                           shuffle_batches=shuffle_batches,
                                                           **kwargs)

## Algos that should not be run in all-algo test because they are not yet finished
WIP_ALGOS = [DynamicsPrediction, InverseDynamicsPrediction]
