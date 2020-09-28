from il_representations.algos.representation_learner import RepresentationLearner, DEFAULT_HARDCODED_PARAMS
from il_representations.algos.encoders import MomentumEncoder, InverseDynamicsEncoder, TargetStoringActionEncoder, \
    RecurrentEncoder, BaseEncoder, VAEEncoder, ActionEncodingEncoder, infer_action_shape_info
from il_representations.algos.decoders import NoOp, MomentumProjectionHead, \
    BYOLProjectionHead, ActionConditionedVectorDecoder, ActionPredictionHead, PixelDecoder, SymmetricProjectionHead, \
    AsymmetricProjectionHead
from il_representations.algos.losses import SymmetricContrastiveLoss, AsymmetricContrastiveLoss, MSELoss, CEBLoss, \
    QueueAsymmetricContrastiveLoss, BatchAsymmetricContrastiveLoss, NegativeLogLikelihood, VAELoss

from il_representations.algos.augmenters import AugmentContextAndTarget, AugmentContextOnly, NoAugmentation
from il_representations.algos.pair_constructors import IdentityPairConstructor, TemporalOffsetPairConstructor
from il_representations.algos.batch_extenders import QueueBatchExtender, IdentityBatchExtender
from il_representations.algos.optimizers import LARS
from il_representations.algos.representation_learner import get_default_args


class SimCLR(RepresentationLearner):
    """
    Implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    https://arxiv.org/abs/2002.05709

    This method works by using a contrastive loss to push together representations of two differently-augmented
    versions of the same image. In particular, it uses a symmetric contrastive loss, which compares the
    (target, context) similarity against similarity of context with all other targets, and also similarity
     of target with all other contexts.
    """
    # TODO note: not made to use momentum because not being used in experiments
    def __init__(self, env, log_dir, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=SymmetricProjectionHead,
                                     loss_calculator=SymmetricContrastiveLoss,
                                     augmenter=AugmentContextAndTarget,
                                     target_pair_constructor=IdentityPairConstructor,
                                     batch_extender=IdentityBatchExtender)

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class TemporalCPC(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        """
        Implementation of a non-recurrent version of CPC: Contrastive Predictive Coding
        https://arxiv.org/abs/1807.03748

        By default, augments only the context, but can be modified to augment both context and target.
        """
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=NoOp,
                                     loss_calculator=QueueAsymmetricContrastiveLoss,
                                     augmenter=NoAugmentation,
                                     batch_extender=IdentityBatchExtender,
                                     target_pair_constructor=TemporalOffsetPairConstructor)
        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(env=env,
                         log_dir=log_dir,
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
        algo_hardcoded_kwargs = dict(shuffle_batches=False,
                                     encoder=RecurrentEncoder,
                                     decoder=SymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=QueueAsymmetricContrastiveLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class MoCo(RepresentationLearner):
    """
    Implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, env, log_dir, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=MomentumEncoder,
                                     decoder=NoOp,
                                     batch_extender=QueueBatchExtender,
                                     augmenter=AugmentContextAndTarget,
                                     loss_calculator=QueueAsymmetricContrastiveLoss,
                                     target_pair_constructor=IdentityPairConstructor)

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class BYOL(RepresentationLearner):
    """Implementation of Bootstrap Your Own Latent: https://arxiv.org/pdf/2006.07733.pdf

    The hyperparameters do _not_ match those in the paper (see Section 3.2).
    Most notably, the momentum weight is set to a constant, instead of annealed
    from 0.996 to 1 via cosine scheduling.
    """
    def __init__(self, env, log_dir, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=MomentumEncoder,
                                     decoder=BYOLProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=AugmentContextAndTarget,
                                     loss_calculator=MSELoss,
                                     target_pair_constructor=IdentityPairConstructor)
        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class CEB(RepresentationLearner):
    """
    CEB with variance that is learned by StochasticEncoder
    """
    def __init__(self, env, log_dir, **kwargs):
        algo_hardcoded_kwargs = dict(encoder_kwargs=dict(stochastic=True),
                                     encoder=BaseEncoder,
                                     decoder=SymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=CEBLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)
        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class FixedVarianceCEB(RepresentationLearner):
    """
    CEB with fixed rather than learned variance
    """
    def __init__(self, env, log_dir, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=SymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=CEBLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class FixedVarianceTargetProjectedCEB(RepresentationLearner):
    """
    CEB with a fixed (rather than learned) variance that also has a projection layer
    to more closely mimic a learned bilinear loss
    """
    def __init__(self, env, log_dir, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=AsymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=CEBLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)
        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


def get_action_representation_dim(action_space, encoder_kwargs):
    """
    Infer what dimension of action representation will be produced by an encoder with the given
    `encoder_kwargs` operating on the given `action_space`.

    :return:
    """
    relevant_encoder_kwargs = get_default_args(ActionEncodingEncoder.__init__)
    relevant_encoder_kwargs.update(encoder_kwargs)

    if relevant_encoder_kwargs['use_lstm']:
        # If the encoder will use a LSTM to encoder the actions, they will be encoded into a vector
        # of size `action_encoding_dim`
        action_representation_dim = relevant_encoder_kwargs['action_encoding_dim']

    else:
        # If the encoder just takes an average over processed action representations, it matters
        # what that processed action representation looks like (e.g. a flattening or an embedding), which
        # is inferred and returned by `infer_action_shape_info`
        action_representation_dim, _, _ = infer_action_shape_info(action_space,
                                                                  relevant_encoder_kwargs['action_embedding_dim'])

    return action_representation_dim


class DynamicsPrediction(RepresentationLearner):
    """
    A basic form of Dynamics modeling, predicting the pixels of the next frame
    given the current frame and the current actions. Uses transposed convolutions
    to do pixel predictions, and a MSE loss to calculate error w.r.t next frame
    """
    def __init__(self, env, log_dir, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        encoder_cls_key = encoder_kwargs.get('obs_encoder_cls', None)

        action_representation_dim = get_action_representation_dim(env.action_space, encoder_kwargs)

        algo_hardcoded_kwargs = dict(encoder=TargetStoringActionEncoder,
                                     decoder=PixelDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=MSELoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     target_pair_constructor_kwargs=dict(mode='dynamics'),
                                     encoder_kwargs=dict(action_space=env.action_space, stochastic=False),
                                     decoder_kwargs=dict(observation_space=env.observation_space,
                                                         encoder_arch_key=encoder_cls_key,
                                                         action_representation_dim=action_representation_dim),
                                     preprocess_extra_context=False)

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class VariationalAutoencoder(RepresentationLearner):
    """
    A basic variational autoencoder that tries to reconstruct the
    current frame, and calculates a VAE loss over current frame pixels,
    using reconstruction loss and a KL divergence between learned
    z distribution and a normal prior
    """
    def __init__(self, env, log_dir, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        encoder_cls_key = encoder_kwargs.get('obs_encoder_cls', None)

        algo_hardcoded_kwargs = dict(encoder=VAEEncoder,
                                     decoder=PixelDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=VAELoss,
                                     target_pair_constructor=IdentityPairConstructor,
                                     decoder_kwargs=dict(observation_space=env.observation_space,
                                                         encoder_arch_key=encoder_cls_key,
                                                         sample=True))

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class InverseDynamicsPrediction(RepresentationLearner):
    """
    An implementation of an inverse dynamics model that tries to predict the action taken
    in between two successive frames, given representations of those frames
    """
    def __init__(self, env, log_dir, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=InverseDynamicsEncoder,
                                     decoder=ActionPredictionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=NegativeLogLikelihood,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     target_pair_constructor_kwargs=dict(mode='inverse_dynamics'),
                                     decoder_kwargs=dict(action_space=env.action_space),
                                     preprocess_target=False)
        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)


class ActionConditionedTemporalVAE(RepresentationLearner):
    """
    Essentially the same as a DynamicsModel, but with learned standard deviation (stochastic=True) and
    VAELoss instead of MSELoss
    """
    def __init__(self, env, log_dir, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        encoder_cls_key = encoder_kwargs.get('obs_encoder_cls', None)

        action_representation_dim = get_action_representation_dim(env.action_space, encoder_kwargs)

        algo_hardcoded_kwargs = dict(encoder=TargetStoringActionEncoder,
                                     decoder=PixelDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=VAELoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     target_pair_constructor_kwargs=dict(mode='dynamics'),
                                     encoder_kwargs=dict(action_space=env.action_space, stochastic=True),
                                     decoder_kwargs=dict(observation_space=env.observation_space,
                                                         encoder_arch_key=encoder_cls_key,
                                                         action_representation_dim=action_representation_dim),
                                     preprocess_extra_context=False)

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(env=env,
                         log_dir=log_dir,
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
    def __init__(self, env, log_dir, **kwargs):
        # Figure out action_encoding_dim. Do this by simply using action_encoding_dim if use_lstm is true,
        # and by calling infer_action_info to get processed_action_dim otherwise
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        action_representation_dim = get_action_representation_dim(env.action_space, encoder_kwargs)

        algo_hardcoded_kwargs = dict(encoder=ActionEncodingEncoder,
                                     decoder=ActionConditionedVectorDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=BatchAsymmetricContrastiveLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     preprocess_extra_context=False,
                                     target_pair_constructor_kwargs=dict(mode='dynamics'),
                                     encoder_kwargs=dict(action_space=env.action_space),
                                     decoder_kwargs=dict(action_representation_dim=action_representation_dim))

        kwargs = self.validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(env=env,
                         log_dir=log_dir,
                         **kwargs)

## Algos that should not be run in all-algo test because they are not yet finished
WIP_ALGOS = []
