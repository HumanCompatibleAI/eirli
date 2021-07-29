from abc import ABC, abstractmethod

import imitation.util.logger as logger
from pyro.distributions import Delta
import stable_baselines3.common.logger as sb_logger
import torch
import torch.nn.functional as F
# losses can be accessed through torch.nn, but I'm doing this to appease PyType
import torch.nn.modules.loss as torch_losses


class RepresentationLoss(ABC):
    def __init__(self, device, sample=False):
        self.device = device
        self.sample = sample

    @abstractmethod
    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist):
        pass

    def get_vector_forms(self, *args):
        return [el.rsample() if self.sample else el.mean for el in args]


class AsymmetricContrastiveLoss(RepresentationLoss):
    """
    A basic contrastive loss that only does prediction/similarity comparison in one direction,
    only calculating a softmax of IJ similarity against all similarities with I.
    """
    def __init__(self, device, sample=False, temp=0.1, normalize=True):
        super(AsymmetricContrastiveLoss, self).__init__(device, sample)
        self.criterion = torch_losses.CrossEntropyLoss()
        self.temp = temp

        # Most methods use either cosine similarity or matrix multiplication similarity. Since cosine similarity equals
        # taking MatMul on normalized vectors, setting normalize=True is equivalent to using torch.CosineSimilarity().
        self.normalize = normalize

        # Sometimes the calculated vectors may contain an image's similarity with itself, which can be a large number.
        # Since we mostly care about maximizing an image's similarity with its augmented version, we subtract a large
        # number to make the classification have ~0 probability picking the original image itself.
        self.large_num = 1e9

    def calculate_logits_and_labels(self, z_i, z_j, mask):
        raise NotImplementedError

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets = self.get_vector_forms(decoded_context_dist, target_dist)

        z_i = decoded_contexts
        z_j = targets

        if self.normalize:  # Use cosine similarity
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        batch_size = z_i.shape[0]
        mask = (torch.eye(batch_size) * self.large_num).to(self.device)

        logits, labels = self.calculate_logits_and_labels(z_i, z_j, mask)
        logits /= self.temp
        return self.criterion(logits, labels)


class QueueAsymmetricContrastiveLoss(AsymmetricContrastiveLoss):
    """
    This implements algorithms that use a queue to maintain all the negative examples. The contrastive loss is
    calculated through the comparison of an image with its augmented version (positive example) and everything else
    in the queue (negative examples). The method used in MoCo.

    Alternatively, for higher sample efficiency, one may use (1) current batch's augmented images, (2) current batch's
    original images, and (3) all the images in the queue as negative examples. This is implemented with setting
    use_batch_neg=True.
    """

    def __init__(self, device, sample=False, temp=0.1, use_batch_neg=False):
        super(QueueAsymmetricContrastiveLoss, self).__init__(device, sample)

        self.temp = temp
        self.use_batch_neg = use_batch_neg  # Use other images in current batch as negative samples

    def calculate_logits_and_labels(self, z_i, z_j, mask):
        """
        z_i: dim (N, C). N - batch_size, C - representation dimension
        z_j: dim (N+K, C). K - number of entries in the queue.
        """
        batch_size = z_i.shape[0]
        queue = z_j[batch_size:]
        z_j = z_j[:batch_size]

        # Calculate the dot product similarity of each image with all images in the queue. Return an NxK tensor.
        l_neg = torch.matmul(z_i, queue.T)  # NxK

        if self.use_batch_neg:
            # Dot product similarity with all other images in the batch
            logits_aa = torch.matmul(z_i, z_i.T)  # NxN

            # Values on the diagonal line are each image's similarity with itself
            logits_aa = logits_aa - mask

            # Dot product similarity with all other augmented images in the batch
            logits_ab = torch.matmul(z_i, z_j.T)

            logits = torch.cat([logits_ab, logits_aa, l_neg], dim=1)  # Nx(2N+K)

            # The values we want to maximize lie on the i-th index of each row i. i.e. the dot product of
            # represent(image_i) and represent(augmented_image_i).
            labels = torch.arange(batch_size, dtype=torch.long).to(self.device)

        else:
            # torch.einsum provides an elegant way to calculate vector dot products across a batch. Each entry on the
            # Nx1 returned tensor is a dot product of represent(image_i) and represent(augmented_image_i).
            l_pos = torch.einsum('nc,nc->n', [z_i, z_j]).unsqueeze(-1)  # Nx1

            # The negative examples here only contain image representations in the queue.
            logits = torch.cat([l_pos, l_neg], dim=1)  # Nx(1+K)

            # The values we want to maximize lie on the 0-th index of each row.
            labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        return logits, labels


class BatchAsymmetricContrastiveLoss(AsymmetricContrastiveLoss):
    """
    This applies to algorithms that performs asymmetric contrast with samples in the same batch. i.e. Negative examples
    come from all other images (and their augmented versions) in the same batch. Represents InfoNCE used in original
    CPC paper.
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(BatchAsymmetricContrastiveLoss, self).__init__(device, sample)
        self.temp = temp

    def calculate_logits_and_labels(self, z_i, z_j, mask):
        """
        z_i: dim (N, C). N - batch_size, C - representation dimension
        z_j: dim (N, C). The augmented images' representations
        """
        batch_size = z_i.shape[0]

        # Similarity of the original images with all other original images in current batch. Return a matrix of NxN.
        logits_aa = torch.matmul(z_i, z_i.T)  # NxN

        # Values on the diagonal line are each image's similarity with itself
        logits_aa = logits_aa - mask

        # Similarity of original images and augmented images
        logits_ab = torch.matmul(z_i, z_j.T)  # NxN

        logits = torch.cat((logits_ab, logits_aa), 1)  # Nx2N

        # The values we want to maximize lie on the i-th index of each row i. i.e. the dot product of
        # represent(image_i) and represent(augmented_image_i).
        label = torch.arange(batch_size, dtype=torch.long).to(self.device)
        return logits, label


class SymmetricContrastiveLoss(RepresentationLoss):
    """
    A contrastive loss that does prediction "in both directions," i.e. that calculates logits of IJ similarity against
    all similarities with J, and also all similarities with I, and calculates cross-entropy on both
    """

    def __init__(self, device, sample=False, temp=0.1, normalize=True):
        super(SymmetricContrastiveLoss, self).__init__(device, sample)

        self.criterion = torch_losses.CrossEntropyLoss()
        self.temp = temp

        # Most methods use either cosine similarity or matrix multiplication similarity. Since cosine similarity equals
        # taking MatMul on normalized vectors, setting normalize=True is equivalent to using torch.CosineSimilarity().
        self.normalize = normalize

        # Sometimes the calculated vectors may contain an image's similarity with itself, which can be a large number.
        # Since we mostly care about maximizing an image's similarity with its augmented version, we subtract a large
        # number to make the classification have ~0 probability picking the original image itself.
        self.large_num = 1e9

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets = self.get_vector_forms(decoded_context_dist, target_dist)
        z_i = decoded_contexts
        z_j = targets
        batch_size = z_i.shape[0]

        if self.normalize:  # Use cosine similarity
            z_i = F.normalize(z_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        mask = (torch.eye(batch_size) * self.large_num).to(self.device)

        # Similarity of the original images with all other original images in current batch. Return a matrix of NxN.
        logits_aa = torch.matmul(z_i, z_i.T)  # NxN

        # Values on the diagonal line are each image's similarity with itself
        logits_aa = logits_aa - mask
        # Similarity of the augmented images with all other augmented images.
        logits_bb = torch.matmul(z_j, z_j.T)  # NxN
        logits_bb = logits_bb - mask
        # Similarity of original images and augmented images
        logits_ab = torch.matmul(z_i, z_j.T)  # NxN
        logits_ba = torch.matmul(z_j, z_i.T)  # NxN

        avg_self_similarity = logits_ab.diag().mean().item()
        logits_other_sim_mask = ~torch.eye(batch_size, dtype=bool, device=logits_ab.device)
        avg_other_similarity = logits_ab.masked_select(logits_other_sim_mask).mean().item()

        sb_logger.record('avg_self_similarity', avg_self_similarity)
        sb_logger.record('avg_other_similarity', avg_other_similarity)
        sb_logger.record('self_other_sim_delta', avg_self_similarity - avg_other_similarity)

        # Each row now contains an image's similarity with the batch's augmented images & original images. This applies
        # to both original and augmented images (hence "symmetric").
        logits_i = torch.cat((logits_ab, logits_aa), 1)  # Nx2N
        logits_j = torch.cat((logits_ba, logits_bb), 1)  # Nx2N
        logits = torch.cat((logits_i, logits_j), axis=0)  # 2Nx2N
        logits /= self.temp

        # The values we want to maximize lie on the i-th index of each row i. i.e. the dot product of
        # represent(image_i) and represent(augmented_image_i).
        label = torch.arange(batch_size, dtype=torch.long).to(self.device)
        labels = torch.cat((label, label), axis=0)

        return self.criterion(logits, labels)


class NegativeLogLikelihood(RepresentationLoss):
    """
    A version of negative log likelihood that directly calculates from distributions
    Uses the mean of target_dist as ground truth, calculates log_prob under
    decoded_context_dist, negates and averages across batch
    """
    def __init__(self, device, sample=False):
        super().__init__(device, sample)

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        assert isinstance(target_dist, Delta), "Target distribution should be a " \
                                               "Delta distribution around a ground truth value"
        # target dist is a Dirac Delta distribution containing the ground truth values
        # decoded_context_dist is a predicted distribution we want to put high probability on the ground truth values

        # Negative log likelihood loss. Using this rather than torch_losses.NLLLoss() so I can work directly
        # with Torch distribution objects
        ground_truth = torch.squeeze(target_dist.mean)
        log_probas = decoded_context_dist.log_prob(ground_truth)
        return torch.mean(-1*log_probas)


class MSELoss(RepresentationLoss):
    """
    A loss that calculates Mean Squared Error between samples or means drawn from
    target_dist and context_dist
    """
    def __init__(self, device, sample=False):
        super().__init__(device, sample)
        self.criterion = torch_losses.MSELoss()

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        decoded_contexts, targets = self.get_vector_forms(decoded_context_dist, target_dist)
        return self.criterion(decoded_contexts, targets)


class CrossEntropyLoss(RepresentationLoss):
    def __init__(self, device, sample=False):
        super().__init__(device, sample)
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, decoded_contexts, targets, encoded_context_dist=None):
        return self.criterion(decoded_contexts, torch.squeeze(targets))


class VAELoss(RepresentationLoss):
    """
    An additive combination of negative log likelihood and
    KL divergence between a Normal distribution prior on z and
    the conditioned-on-x z distribution

    Note that beta of 1e-6 gives ~perfect autoencoding on finger-spin after
    10,000 batches. Pushing it up to 1e-5 makes it slightly noisier. Starts to
    struggle around 1e-4, and doesn't learn anything around 1e-3.
    """
    def __init__(self, device, sample=False, beta=1e-5, prior_scale=1.0):
        super().__init__(device, sample)
        self.beta = beta  # The relative weight on the KL Divergence/regularization loss, relative to reconstruction
        self.prior_scale = prior_scale  # The scale parameter used to construct prior used in KLD

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        (ground_truth_pixels,) = self.get_vector_forms(target_dist)
        predicted_pixels = decoded_context_dist.mean

        recon_loss = F.mse_loss(predicted_pixels, ground_truth_pixels)

        prior = torch.distributions.Normal(torch.zeros(encoded_context_dist.batch_shape +
                                                       encoded_context_dist.event_shape).to(self.device),
                                           self.prior_scale)
        independent_prior = torch.distributions.Independent(prior,
                                                            len(encoded_context_dist.event_shape))
        kld = torch.distributions.kl.kl_divergence(encoded_context_dist, independent_prior)

        logger.record('loss_recon', recon_loss.item())
        logger.record('loss_kld', torch.mean(kld).item())

        loss = recon_loss + self.beta * torch.mean(kld)
        return loss


class AELoss(RepresentationLoss):
    """
    Compute the reconstruction (MSE) loss between the generated image and the original image.
    """
    def __init__(self, device, sample=False):
        super().__init__(device, sample)

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        (ground_truth_pixels,) = self.get_vector_forms(target_dist)
        predicted_pixels = decoded_context_dist.mean

        loss = F.mse_loss(predicted_pixels, ground_truth_pixels)
        return loss


class CEBLoss(RepresentationLoss):
    """
    A variational contrastive loss that implements information bottlenecking, but in a less conservative form
    than done by traditional VIB techniques
    """

    def __init__(self, device, beta=.1, sample=True):
        super().__init__(device, sample=sample)
        # TODO allow for beta functions
        self.beta = beta
        self.sample = sample

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        z = decoded_context_dist.rsample()

        log_ezx = decoded_context_dist.log_prob(z) # B -> Log proba of each vector in z under the distribution it was sampled from
        log_bzy = target_dist.log_prob(z) # B -> Log proba of each vector in z under the distribution conditioned on its corresponding target
        cross_probas_logits = torch.stack([target_dist.log_prob(z[i]) for i in range(z.shape[0])], dim=0) # BxB Log proba of each vector z[i] under _all_ target distributions
        # The return shape of target_dist.log_prob(z[i]) is the probability of z[i] under each distribution in the batch
        catgen = torch.distributions.Categorical(logits=cross_probas_logits) # logits of shape BxB -> Batch categorical, one distribution per element in z over possible
                                                                      # targets/y values
        inds = (torch.arange(start=0, end=len(z))).to(self.device)
        i_yz = catgen.log_prob(inds) # The probability of the kth target under the kth Categorical distribution (probability of true y)
        loss = torch.mean(self.beta*(log_ezx - log_bzy) - i_yz)
        return loss


class GaussianPriorLoss(RepresentationLoss):
    """
    KL divergence between a Normal distribution prior on z and
    the conditioned-on-x z distribution
    """
    def __init__(self, device, sample=False, prior_scale=1.0):
        super().__init__(device, sample)
        self.prior_scale = prior_scale  # The scale parameter used to construct prior used in KLD

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        prior = torch.distributions.Normal(torch.zeros(encoded_context_dist.batch_shape +
                                                       encoded_context_dist.event_shape).to(self.device),
                                           self.prior_scale)
        independent_prior = torch.distributions.Independent(prior,
                                                            len(encoded_context_dist.event_shape))
        kld = torch.distributions.kl.kl_divergence(encoded_context_dist, independent_prior)

        loss = torch.mean(kld)
        return loss
