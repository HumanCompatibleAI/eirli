.. _rep-learner-design:


Design Principles
=================

The design of this repo's core RepresentationLearner abstraction was based around a deconstruction of common RepL learners
into their component parts in a way that we feel strikes a good balance between flexibility and reuseability.

To explain a bit more about the different components, let's look at four algorithms: a VAE, a Temporal VAE, SimCLR, and Temporal Contrastive Predictive Coding (CPC).
What are the differences between different pairs of these?

- A VAE and TemporalVAE function basically the same way, except that, instead of your reconstruction target being the
  same as the input frame, it's one frame forward in a trajectory
- Between a VAE and SimCLR, the former tries to reconstruct an input frame after a bottleneck, and the latter tries to
  achieve similarity with the representation of a differently-augmented input frame after a bottleneck + projection layer. So
  between these two algorithms, you can identify the differences of (1) using augmentation vs not, and (2) using a contrastive
  loss rather than a reconstructive one. However, they're the same insofar as the "target" in both cases is (some modification of)
  the input frame itself. This same analogy holds between TemporalVAE and TemporalCPC: one is reconstructive, one contrastive,
  but both use a temporally-offset target
- Between SimCLR and TemporalCPC, the central difference is that, instead of calculating a contrastive loss between
  augmented versions of the same frame, TemporalCPC calculates a contrastive loss between an augmented frame_t and an augmented
  frame t+k


Now that you've got some practice deconstructing algorithms this way, it may be easier to follow the deconstruction we chose for this codebase.
At the most general level, we define representation learners as following the pattern of:

::
    L = Loss(Decoder(Encoder(Context), OptionalExtraContext)), Decoder(Encoder(Target))

In our framework, different learners are differentiated from one another by their different implementations of each of
these components.

1. First, we take in a dataset and construct Context, Target pairs from it. This is done by a `TargetPairConstructor` object.
   The most common strategies for constructing pairs are  identity (where context and target are the same frame)
   or temporal offset (where context and target are temporally-offset frames). However, there are also situations
   where the target is not an image input, for example, when we want to predict an action, in the case of
   inverse dynamics (ID). In that case, `target` is the action vector. `context` objects are always image frames.
   We also need to handle the case where we need two forms of input information to predict the target: for example,
   predicting `action` given two contiguous frames in ID, or predicting next frame given current frame and action,
   in a Dynamics model. These are stored in an optional `extra_context` object, which some encoders
   have logic to deal with, but which others ignore

2. We augment our context frame (and optionally our target) according to some strategy defined by an `Augmenter`. This is
   a fairly simple process, and the main variation here is (a) whether to augment both context and target or just context,
   and (b) what augmentations, if any, to apply.

3. Then, we take our possibly-augmented dataset of Context, Target pairs and run a batch through the `Encoder`.
   The job of the encoder is to map a context (and optionally also a target) into a `z` representation vector.
   This component is what we transfer to downstream models.

4. In some cases, we need to have a `Decoder` to do postprocessing on the representation learned by the encoder, before
   it is passed to the loss. This component is dropped after RepL training, and not used in downstream finetuning. In the case
   of a contrastive loss with a projection head, this might be a simple MLP. Or, in the case of a VAE, where loss is calculated
   on a reconstructed image, this may be a more complex network to reconstruct an image from a bottlenecked representation.
   Sometimes, it uses `extra_context`, in addition to `context`, to construct an input that can be given to the loss function,
   as in the cases of Dynamics and Inverse Dynamics mentioned above, where you want to use action vector or next frame
   respectively as part of the prediction of the other quantity.
   A decoder may also simply be the identity, in cases where no projection head is used.

5. Once we have run our batches through the encoding and decoding processes, it's time to calculate a loss, with a
   `RepresentationLoss` object. This loss takes the decoded context, decoded target, and sometimes the encoded
   context as input. (The latter is basically only used in the case of VAE, where part of our loss is pulling
   the `p(z|x)` distribution closer to a Gaussian prior).


Given these components, let's compare a few of the definitions of algorithms we gave as examples above.


::

    class VariationalAutoencoder(RepresentationLearner):
        """
        A basic variational autoencoder that tries to reconstruct the
        current frame, and calculates a VAE loss over current frame pixels,
        using reconstruction loss and a KL divergence between learned
        z distribution and a normal prior
        """
        def __init__(self, **kwargs):
            # ... <repeated machinery> ...
            algo_hardcoded_kwargs = dict(encoder=VAEEncoder,
                                         decoder=PixelDecoder,
                                         batch_extender=IdentityBatchExtender,
                                         augmenter=NoAugmentation,
                                         loss_calculator=VAELoss,
                                         target_pair_constructor=IdentityPairConstructor,
                                         decoder_kwargs=dict(observation_space=kwargs['observation_space'],
                                                             encoder_arch_key=dec_encoder_cls_key,
                                                             sample=True))