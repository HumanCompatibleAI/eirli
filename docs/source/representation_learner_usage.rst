.. _rep-learner-usage:


Representation Learner Usage
=============================

All implementations of representation learning algorithms in this codebase are built around one central abstraction,
which is, perhaps not surprisingly, a class called `RepresentationLearner`. This class defines the
general framework by which components of representation learner training happen. Different variants of
RepL algorithms are designed by creating a learner which takes in and runs different implementations of those component steps.


.. _pre_defined:

Training a Pre-Defined Representation Learner
---------------------------------------------

For the sake of our experimentation, we have defined a number of existing, commonly used algorithms, and have made
them available to import and use directly.

::

    from il_representations.algos import SimCLR, NoAugmentation
    from il_representations.utils import convert_to_simple_webdataset, load_simple_webdataset

    # This step converts a Pytorch dataset of the form [{'obs': <image>}...] into a Webdataset
    # that can stream from disk. This step only needs to be performed once
    full_wds_url = convert_to_simple_webdataset(dataset=pytorch_dataset,
                                                file_out_path="temp",
                                                file_out_name="my_dataset")

    wds = load_simple_webdataset(full_wds_url)

    # For this example, we're imagining a (3, 64, 64) image size
    algo = algos.SimCLR(batch_size=10,
                        observation_space=spaces.Box(shape=(3, 64, 64),
                                                     low=0,
                                                     high=1),
                        action_space=None,
                        augmenter=NoAugmentation)
    # This trains for a single epoch of 10 batches, calculating logging
    # information and logging it every step. This is likely substantially
    # more logging than you'd want for a typical training use case
    algo.learn(datasets=[wds], batches_per_epoch=10, n_epochs=1,
               log_dir='temp', log_interval=1, calc_log_interval=1)


.. _define_new:

Defining a New Representation Learner
-------------------------------------
Let's use the SimCLR example from above to walk through how you might create an algorithm that differs from it in some way.
This is the code used in `algos/__init__.py` to define the SimCLR algorithm class, with some additional
explanatory documentation added in for clarity's sake. This explanation will assume you have some familiarity with the
conceptual breakdown used in this codebase; if you're unsure about that, you can read more :ref:`here! <rep-learner-design>`

::

    class SimCLR(RepresentationLearner):
    """
    Implementation of SimCLR: A Simple Framework for
    Contrastive Learning of Visual Representations
    https://arxiv.org/abs/2002.05709

    This method works by using a contrastive loss to push together representations
    of two differently-augmented versions of the same image. In particular, it
    uses a symmetric contrastive loss, which compares the (target, context)
    similarity against similarity of context with all other targets, and also
    similarity of target with all other contexts.
    """
    def __init__(self, **kwargs):
        # This is where we specify the RepresentationLearner arguments
        # that are integral to the algorithm definition of SimCLR

        algo_hardcoded_kwargs = dict(# We use our BaseEncoder to map from image to representation
                                     # The output of `encoder` is what we use for transfer
                                     encoder=BaseEncoder,
                                     # A MLP projection head, symmetric between context and target
                                     # The output of `decoder` is passed to the loss function
                                     decoder=SymmetricProjectionHead,
                                     # A contrastive loss where we try to predict
                                     # target given context and also context given target
                                     loss_calculator=SymmetricContrastiveLoss,
                                     # Augment both context and target before encoder
                                     augmenter=AugmentContextAndTarget,
                                     # For SimCLR, the target and context are different
                                     # augmentations of the same, "Identity" frame
                                     target_pair_constructor=IdentityPairConstructor,
                                     # Since we're not using momentum here, the encoder
                                     # batch is the same as the one used in the loss
                                     batch_extender=IdentityBatchExtender)

        kwargs = validate_and_update_kwargs(kwargs, algo_hardcoded_kwargs=algo_hardcoded_kwargs)

        super().__init__(**kwargs)

Existing Pre-Defined Representation Learners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can find a list of existing algorithms at: :mod:`il_representations.algos`
