from .representation_learner import RepresentationLearner
from .encoders import MomentumEncoder, InverseDynamicsEncoder, DynamicsEncoder, RecurrentEncoder, StochasticEncoder, DeterministicEncoder
from .decoders import ProjectionHead, NoOp, MomentumProjectionHead, BYOLProjectionHead
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
                                     encoder=DeterministicEncoder,
                                     decoder=ProjectionHead,
                                     loss_calculator=SymmetricContrastiveLoss,
                                     augmenter=AugmentContextAndTarget,
                                     target_pair_constructor=IdentityPairConstructor,
                                     **kwargs)


class TemporalCPC(RepresentationLearner):
    """
    Implementation of a non-recurrent version of CPC: Contrastive Predictive Coding
    https://arxiv.org/abs/1807.03748

    By default, augments only the context, but can be modified to augment both context and target.
    """
    def __init__(self, env, log_dir, **kwargs):
        super(TemporalCPC, self).__init__(env=env,
                                          log_dir=log_dir,
                                          encoder=DeterministicEncoder,
                                          decoder=NoOp,
                                          loss_calculator=AsymmetricContrastiveLoss,
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
        super(MoCo, self).__init__(env=env,
                                   log_dir=log_dir,
                                   encoder=MomentumEncoder,
                                   decoder=NoOp,
                                   loss_calculator=AsymmetricContrastiveLoss,
                                   augmenter=AugmentContextAndTarget,
                                   target_pair_constructor=TemporalOffsetPairConstructor,
                                   batch_extender=QueueBatchExtender,
                                   batch_extender_kwargs={'queue_size': queue_size},
                                   **kwargs)


class MoCoWithProjection(RepresentationLearner):
    """
    Implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
    https://arxiv.org/abs/1911.05722

    Includes an additional projection head atop the representation and before the prediction
    """

    def __init__(self, env, log_dir, queue_size=8192, **kwargs):
        super(MoCoWithProjection, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=MomentumEncoder,
                                                 decoder=MomentumProjectionHead,
                                                 loss_calculator=AsymmetricContrastiveLoss,
                                                 augmenter=AugmentContextAndTarget,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 batch_extender=QueueBatchExtender,
                                                 batch_extender_kwargs={'queue_size': queue_size},
                                                 **kwargs)


class DynamicsPrediction(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        super(DynamicsPrediction, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=DynamicsEncoder,
                                                 decoder=NoOp, # Should be a pixel decoder that takes in action, currently errors
                                                 loss_calculator=MSELoss,
                                                 augmenter=AugmentContextAndTarget,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 target_pair_constructor_kwargs={'mode': 'dynamics'},
                                                 **kwargs)


class InverseDynamicsPrediction(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        super(InverseDynamicsPrediction, self).__init__(env=env,
                                                        log_dir=log_dir,
                                                        encoder=InverseDynamicsEncoder,
                                                        decoder=NoOp, # Should be a action decoder that takes in next obs representation
                                                        loss_calculator=MSELoss,
                                                        augmenter=AugmentContextAndTarget,
                                                        target_pair_constructor=TemporalOffsetPairConstructor,
                                                        target_pair_constructor_kwargs={'mode': 'inverse_dynamics'},
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

## Algos that should not be run in all-algo test because they are not yet finished
WIP_ALGOS = [DynamicsPrediction, InverseDynamicsPrediction]
