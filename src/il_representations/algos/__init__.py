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


def update_kwarg_dict(kwargs, kwargs_key, update_dict, cls):
    """
    Updates an internal kwargs dict within `kwargs`, specified by `kwargs_key`, to
    contain the values within `update_dict`

    :param kwargs: A dictionary for all RepresentationLearner kwargs
    :param kwargs_key: A key indexing into `kwargs` representing a keyword arg that is itself a kwargs dictionary.
    :param update_dict: A dict containing the key/value changes that should be made to kwargs[kwargs_key]
    :param cls: The class on which this is being called
    :return:
    """
    internal_kwargs = kwargs.get(kwargs_key) or {}
    for key, value in update_dict.items():
        if key in internal_kwargs:
            assert internal_kwargs[key] == value, f"{cls.__name__} tried to directly set keyword arg {key} to {value}, but it was specified elsewhere as {kwargs[key]}"
            raise Warning(f"In {cls.__name__}, {key} was specified as both a direct argument and in a kwargs dictionary. Prefer using only one for robustness reasons.")
        internal_kwargs[key] = value

    kwargs[kwargs_key] = internal_kwargs


def clean_kwargs(kwargs, cls, keys=None):
    """
    Checks to confirm that you're not passing in an non-default value for a parameter that gets hardcoded
    by the class definition

    :param kwargs: Dictionary of all RepresentationLearner params
    :param cls: The class on which this is being called
    :param keys: The keys that are hardcoded by the class definition
    :return:
    """
    default_args = get_default_args(RepresentationLearner.__init__)
    if keys is None:
        keys = DEFAULT_HARDCODED_PARAMS
    for k in keys:
        if k not in kwargs:
            continue
        assert kwargs[k] == default_args[k], f"You passed in a non-default value for parameter {k} hardcoded by {cls.__name__}"
        del kwargs[k]
    return kwargs


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
        kwargs = clean_kwargs(kwargs, self.__class__)

        super(SimCLR, self).__init__(env=env,
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
        kwargs = clean_kwargs(kwargs, self.__class__)

        update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                          {'temporal_offset': temporal_offset}, self.__class__)
        super(TemporalCPC, self).__init__(env=env,
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
        kwargs = clean_kwargs(kwargs, self.__class__)
        kwargs['shuffle_batches'] = False
        super(RecurrentCPC, self).__init__(env=env,
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
        kwargs = clean_kwargs(kwargs, self.__class__, hardcoded_params)
        update_kwarg_dict(kwargs, 'batch_extender_kwargs', {'queue_size': queue_size}, self.__class__)
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
        kwargs = clean_kwargs(kwargs, self.__class__, hardcoded_params)

        update_kwarg_dict(kwargs, 'batch_extender_kwargs', {'queue_size': queue_size},
                                                  self.__class__)
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
        kwargs = clean_kwargs(kwargs, self.__class__)

        update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                                                           {'mode': 'dynamics'}, self.__class__)
        super(DynamicsPrediction, self).__init__(env=env,
                                                 log_dir=log_dir,
                                                 encoder=DynamicsEncoder,
                                                 decoder=NoOp, # Should be a pixel decoder that takes in action, currently errors
                                                 loss_calculator=MSELoss,
                                                 augmenter=AugmentContextOnly,
                                                 target_pair_constructor=TemporalOffsetPairConstructor,
                                                 **kwargs)


class InverseDynamicsPrediction(RepresentationLearner):
    def __init__(self, env, log_dir, **kwargs):
        kwargs = clean_kwargs(kwargs, self.__class__)
        update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                                                           {'mode': 'inverse_dynamics'}, self.__class__)
        super(InverseDynamicsPrediction, self).__init__(env=env,
                                                        log_dir=log_dir,
                                                        encoder=InverseDynamicsEncoder,
                                                        decoder=NoOp, # Should be a action decoder that takes in next obs representation
                                                        loss_calculator=MSELoss,
                                                        augmenter=AugmentContextOnly,
                                                        target_pair_constructor=TemporalOffsetPairConstructor,
                                                        **kwargs)


class BYOL(RepresentationLearner):
    """Implementation of Bootstrap Your Own Latent: https://arxiv.org/pdf/2006.07733.pdf

    The hyperparameters do _not_ match those in the paper (see Section 3.2).
    Most notably, the momentum weight is set to a constant, instead of annealed
    from 0.996 to 1 via cosine scheduling.
    """
    def __init__(self, env, log_dir, **kwargs):
        kwargs = clean_kwargs(kwargs, self.__class__)
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
        kwargs = clean_kwargs(kwargs, self.__class__)
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
        kwargs = clean_kwargs(kwargs, self.__class__)
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
        kwargs = clean_kwargs(kwargs, self.__class__)
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
        kwargs = clean_kwargs(kwargs, self.__class__)
        kwargs['preprocess_extra_context'] = False
        update_kwarg_dict(kwargs, 'target_pair_constructor_kwargs',
                                                           {"temporal_offset": temporal_offset, "mode": "dynamics"}, self.__class__)

        update_kwarg_dict(kwargs, 'decoder_kwargs',
                                           {'action_space': env.action_space}, self.__class__)

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
