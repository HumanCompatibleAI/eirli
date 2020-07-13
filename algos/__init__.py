from .representation_learner import RepresentationLearner
from .encoders import CNNEncoder, MomentumEncoder, InverseDynamicsEncoder, DynamicsEncoder
from .decoders import ProjectionHead, NoOp, MomentumProjectionHead, LSTMHead
from .losses import SymmetricContrastiveLoss, AsymmetricContrastiveLoss, MSELoss
from .augmenters import AugmentContextAndTarget, AugmentContextOnly
from .pair_constructors import IdentityPairConstructor, TemporalOffsetPairConstructor
from .batch_extenders import QueueBatchExtender
from .optimizers import LARS


class SimCLR(RepresentationLearner):
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
    def __init__(self, env, log_dir, **kwargs):
        super(TemporalCPC, self).__init__(env=env,
                                          log_dir=log_dir,
                                          encoder=CNNEncoder,
                                          decoder=NoOp,
                                          loss_calculator=AsymmetricContrastiveLoss,
                                          augmenter=AugmentContextOnly,
                                          target_pair_constructor=TemporalOffsetPairConstructor,
                                          **kwargs)


# NB these optimizer_kwargs are copy-pasted, and thus may not be sensible for MoCo
# Currently breaks due to batch sizes not being the same each run (sometimes < 256)
class MoCo(RepresentationLearner):
    def __init__(self, env, log_dir, queue_size=8192, **kwargs):
        super(MoCo, self).__init__(env=env,
                                   log_dir=log_dir,
                                   encoder=MomentumEncoder,
                                   decoder=NoOp,
                                   loss_calculator=AsymmetricContrastiveLoss,
                                   augmenter=AugmentContextAndTarget,
                                   target_pair_constructor=TemporalOffsetPairConstructor,
                                   batch_extender=QueueBatchExtender,
                                   queue_size=queue_size,
                                   **kwargs)


class MoCoWithProjection(RepresentationLearner):
    def __init__(self, env, log_dir, queue_size=8192, **kwargs):
        super(MoCoWithProjection, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=MomentumEncoder,
                                                 decoder=MomentumProjectionHead,
                                                 loss_calculator=AsymmetricContrastiveLoss,
                                                 augmenter=AugmentContextAndTarget,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 batch_extender=QueueBatchExtender,
                                                 queue_size=queue_size,
                                                 **kwargs)


class RNNTest(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        super(RNNTest, self).__init__(env=env,
                                      log_dir=log_dir,
                                      encoder=CNNEncoder,
                                      decoder=LSTMHead,
                                      loss_calculator=AsymmetricContrastiveLoss,
                                      augmenter=AugmentContextAndTarget,
                                      target_pair_constructor=IdentityPairConstructor,
                                      recurrent=True,
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