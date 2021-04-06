from il_representations.algos.representation_learner import RepresentationLearner, DEFAULT_HARDCODED_PARAMS
from il_representations.algos.encoders import MomentumEncoder, InverseDynamicsEncoder, TargetStoringActionEncoder, \
    RecurrentEncoder, BaseEncoder, VAEEncoder, JigsawEncoder, ActionEncodingEncoder, infer_action_shape_info, \
    ActionEncodingInverseDynamicsEncoder
from il_representations.algos.decoders import NoOp, MomentumProjectionHead, \
    BYOLProjectionHead, ActionConditionedVectorDecoder, ContrastiveInverseDynamicsConcatenationHead, \
    ActionPredictionHead, PixelDecoder, SymmetricProjectionHead, AsymmetricProjectionHead, JigsawProjectionHead
from il_representations.algos.losses import SymmetricContrastiveLoss, AsymmetricContrastiveLoss, MSELoss, CEBLoss, \
    QueueAsymmetricContrastiveLoss, BatchAsymmetricContrastiveLoss, NegativeLogLikelihood, VAELoss, AELoss, \
    CrossEntropyLoss
from il_representations.algos.augmenters import AugmentContextAndTarget, AugmentContextOnly, NoAugmentation, \
    AugmentContextAndExtraContext
from il_representations.algos.pair_constructors import IdentityPairConstructor, TemporalOffsetPairConstructor, \
    JigsawPairConstructor
from il_representations.algos.batch_extenders import QueueBatchExtender, IdentityBatchExtender
from il_representations.algos.optimizers import LARS
from il_representations.algos.representation_learner import get_default_args
import logging


def validate_and_update_kwargs(user_kwargs, algo_hardcoded_kwargs):
    # return a copy instead of updating in-place to avoid inconsistent state
    # after a failed update
    merged_kwargs = user_kwargs.copy()

    # Check if there are algo_hardcoded_kwargs that we need to add to the user kwarg s
    for param_name, param_value in algo_hardcoded_kwargs.items():
        # If there's a shared dict param in both user_kwargs and algo_hardcoded_kwargs,
        # recursively call this method to merge them together
        if param_name in merged_kwargs and isinstance(merged_kwargs[param_name], dict):
            merged_kwargs[param_name] = validate_and_update_kwargs(merged_kwargs[param_name], param_value)
        # If there's a shared non-dict param, warn that a param hardcoded by the algorithm
        # will be ignored and the user-input param used instead
        elif param_name in merged_kwargs and merged_kwargs[param_name] != param_value:
            logging.warning(
                f"Overwriting algorithm-hardcoded value {param_value} of "
                f"param {param_name} with user value {merged_kwargs[param_name]}")
        # If there's no competing param in user-passed kwargs, add the hardcoded key and value
        # to the merge kwargs dict
        else:
            merged_kwargs[param_name] = algo_hardcoded_kwargs[param_name]
    return merged_kwargs


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
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=SymmetricProjectionHead,
                                     loss_calculator=SymmetricContrastiveLoss,
                                     augmenter=AugmentContextAndTarget,
                                     target_pair_constructor=IdentityPairConstructor,
                                     batch_extender=IdentityBatchExtender)

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(**kwargs)


class TemporalCPC(RepresentationLearner):
    def __init__(self, **kwargs):
        """
        Implementation of a non-recurrent version of CPC: Contrastive Predictive Coding
        https://arxiv.org/abs/1807.03748

        """
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=NoOp,
                                     loss_calculator=BatchAsymmetricContrastiveLoss,
                                     augmenter=AugmentContextAndTarget,
                                     batch_extender=IdentityBatchExtender,
                                     target_pair_constructor=TemporalOffsetPairConstructor)
        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(**kwargs)


class RecurrentCPC(RepresentationLearner):
    """
    Implementation of a recurrent version of CPC: Contrastive Predictive Coding
    https://arxiv.org/abs/1807.03748

    The encoder first encodes individual frames for both context and target, and then, for the context,
    builds up a recurrent representation of all prior frames in the same trajectory, to use to predict the target.

    By default, augments only the context, but can be modified to augment both context and target.
    """
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(shuffle_batches=False,
                                     encoder=RecurrentEncoder,
                                     decoder=SymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=QueueAsymmetricContrastiveLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class MoCo(RepresentationLearner):
    """
    Implementation of MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=MomentumEncoder,
                                     decoder=NoOp,
                                     batch_extender=QueueBatchExtender,
                                     augmenter=AugmentContextAndTarget,
                                     loss_calculator=QueueAsymmetricContrastiveLoss,
                                     target_pair_constructor=IdentityPairConstructor)

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class BYOL(RepresentationLearner):
    """Implementation of Bootstrap Your Own Latent: https://arxiv.org/pdf/2006.07733.pdf

    The hyperparameters do _not_ match those in the paper (see Section 3.2).
    Most notably, the momentum weight is set to a constant, instead of annealed
    from 0.996 to 1 via cosine scheduling.
    """
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=MomentumEncoder,
                                     decoder=BYOLProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=AugmentContextAndTarget,
                                     loss_calculator=MSELoss,
                                     target_pair_constructor=IdentityPairConstructor)
        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)

    def learn(self, dataset, training_epochs):
        raise NotImplementedError("BYOL decoder currently has an unfixed issue with gradients being None ")


class CEB(RepresentationLearner):
    """
    CEB with variance that is learned by StochasticEncoder
    """
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(encoder_kwargs=dict(learn_scale=False),
                                     decoder_kwargs=dict(learn_scale=True),
                                     encoder=BaseEncoder,
                                     decoder=SymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=CEBLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)
        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class FixedVarianceCEB(RepresentationLearner):
    """
    CEB with fixed rather than learned variance
    """
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=SymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=CEBLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class FixedVarianceTargetProjectedCEB(RepresentationLearner):
    """
    CEB with a fixed (rather than learned) variance that also has a projection layer
    to more closely mimic a learned bilinear loss
    """
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=BaseEncoder,
                                     decoder=AsymmetricProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=CEBLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor)
        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


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
    def __init__(self, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        encoder_cls_key = encoder_kwargs.get('obs_encoder_cls', None)

        action_representation_dim = get_action_representation_dim(kwargs['action_space'], encoder_kwargs)

        algo_hardcoded_kwargs = dict(encoder=TargetStoringActionEncoder,
                                     decoder=PixelDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=MSELoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     target_pair_constructor_kwargs=dict(mode='dynamics'),
                                     encoder_kwargs=dict(action_space=kwargs['action_space'], learn_scale=False),
                                     decoder_kwargs=dict(observation_space=kwargs['observation_space'],
                                                         encoder_arch_key=encoder_cls_key,
                                                         action_representation_dim=action_representation_dim),
                                     preprocess_extra_context=False)

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class VariationalAutoencoder(RepresentationLearner):
    """
    A basic variational autoencoder that tries to reconstruct the
    current frame, and calculates a VAE loss over current frame pixels,
    using reconstruction loss and a KL divergence between learned
    z distribution and a normal prior
    """
    def __init__(self, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        encoder_cls_key = encoder_kwargs.get('obs_encoder_cls', None)

        algo_hardcoded_kwargs = dict(encoder=VAEEncoder,
                                     decoder=PixelDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=VAELoss,
                                     target_pair_constructor=IdentityPairConstructor,
                                     decoder_kwargs=dict(observation_space=kwargs['observation_space'],
                                                         encoder_arch_key=encoder_cls_key,
                                                         sample=True))

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class Jigsaw(RepresentationLearner):
    """
    A basic Jigsaw task that requires a network to solve Jigsaw puzzles.
    """
    def __init__(self, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        encoder_cls_key = encoder_kwargs.get('obs_encoder_cls', None)

        algo_hardcoded_kwargs = dict(encoder=JigsawEncoder,
                                     decoder=JigsawProjectionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=CrossEntropyLoss,
                                     target_pair_constructor=JigsawPairConstructor,
                                     encoder_kwargs=dict(obs_encoder_cls_kwargs={'contain_fc_layer': False}),
                                     decoder_kwargs=dict(architecture=[{'output_dim': 128},
                                                                       {'output_dim': 127}]))

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class InverseDynamicsPrediction(RepresentationLearner):
    """
    An implementation of an inverse dynamics model that tries to predict the action taken
    in between two successive frames, given representations of those frames
    """
    def __init__(self, **kwargs):
        algo_hardcoded_kwargs = dict(encoder=InverseDynamicsEncoder,
                                     decoder=ActionPredictionHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=NegativeLogLikelihood,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     target_pair_constructor_kwargs=dict(mode='inverse_dynamics'),
                                     decoder_kwargs=dict(action_space=kwargs['action_space']),
                                     preprocess_target=False)
        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(**kwargs)


class ContrastiveInverseDynamicsPrediction(RepresentationLearner):
    """
    Like InverseDynamicsPrediction above, except it uses a contrastive loss function.

    A contrastive loss here means we predict a representation of the action given the
    current and next state, and try to match that against the embedding of the true
    action, with negatives provided by the embedding of other actions in the batch.

    During the decoder stage, instead of predicting an action, we simply concatenate
    the representations of s and s' together, and then predict an action representation
    from the concatenation using a projection head. During the encoder stage, we need to
    also encode the actions.

    """
    def __init__(self, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        action_representation_dim = get_action_representation_dim(kwargs['action_space'], encoder_kwargs)
        algo_hardcoded_kwargs = dict(encoder=ActionEncodingInverseDynamicsEncoder,
                                     decoder=ContrastiveInverseDynamicsConcatenationHead,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=AugmentContextAndExtraContext,
                                     loss_calculator=BatchAsymmetricContrastiveLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     target_pair_constructor_kwargs=dict(mode='inverse_dynamics'),
                                     encoder_kwargs=dict(action_space=kwargs['action_space']),
                                     # By default, have the bare minimum projection: a linear layer
                                     decoder_kwargs=dict(projection_architecture=[]),
                                     preprocess_target=False,
                                     projection_dim=action_representation_dim)
        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(**kwargs)


class ActionConditionedTemporalVAE(RepresentationLearner):
    """
    Essentially the same as a DynamicsModel, but with learned standard deviation (learn_scale=True) and
    VAELoss instead of MSELoss
    """
    def __init__(self, **kwargs):
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        encoder_cls_key = encoder_kwargs.get('obs_encoder_cls', None)

        action_representation_dim = get_action_representation_dim(kwargs['action_space'], encoder_kwargs)

        algo_hardcoded_kwargs = dict(encoder=TargetStoringActionEncoder,
                                     decoder=PixelDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=NoAugmentation,
                                     loss_calculator=VAELoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     target_pair_constructor_kwargs=dict(mode='dynamics'),
                                     encoder_kwargs=dict(action_space=kwargs['action_space'], learn_scale=True),
                                     decoder_kwargs=dict(observation_space=kwargs['observation_space'],
                                                         encoder_arch_key=encoder_cls_key,
                                                         action_representation_dim=action_representation_dim),
                                     preprocess_extra_context=False)

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)
        super().__init__(**kwargs)


class ActionConditionedTemporalCPC(RepresentationLearner):
    """
    Implementation of reinforcement-learning-specific variant of Temporal CPC which adds a projection layer on top
    of the learned representation which integrates an encoding of the actions taken between time (t) and whatever
    time (t+k) is specified in temporal_offset and used for pulling out the target frame. This, notionally, allows
    the algorithm to construct frame representations that are action-independent, rather than marginalizing over an
    expected policy, as might need to happen if the algorithm needed to predict the frame at time (t+k) over any
    possible action distribution.
    """
    def __init__(self, **kwargs):
        # Figure out action_encoding_dim. Do this by simply using action_encoding_dim if use_lstm is true,
        # and by calling infer_action_info to get processed_action_dim otherwise
        encoder_kwargs = kwargs.get('encoder_kwargs') or {}
        action_representation_dim = get_action_representation_dim(kwargs['action_space'], encoder_kwargs)

        algo_hardcoded_kwargs = dict(encoder=ActionEncodingEncoder,
                                     decoder=ActionConditionedVectorDecoder,
                                     batch_extender=IdentityBatchExtender,
                                     augmenter=AugmentContextAndTarget,
                                     loss_calculator=BatchAsymmetricContrastiveLoss,
                                     target_pair_constructor=TemporalOffsetPairConstructor,
                                     preprocess_extra_context=False,
                                     target_pair_constructor_kwargs=dict(mode='dynamics'),
                                     encoder_kwargs=dict(action_space=kwargs['action_space'],
                                                         learn_scale=False),
                                     decoder_kwargs=dict(action_representation_dim=action_representation_dim,
                                                         learn_scale=False))

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(**kwargs)

## Algos that should not be run in all-algo test because they are not yet finished
WIP_ALGOS = [BYOL]
