from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent


class RepresentationLoss(ABC):
    def __init__(self, device, sample=False):
        self.device = device
        self.sample = sample

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist):
        pass

    def get_vector_forms(self, decoded_context_dist, target_dist, encoded_context_dist):
        decoded_contexts = decoded_context_dist.sample() if self.sample else decoded_context_dist.loc
        targets = target_dist.sample() if self.sample else target_dist.loc
        encoded_contexts = encoded_context_dist.sample() if self.sample else encoded_context_dist.loc
        return decoded_contexts, targets, encoded_contexts


class AsymmetricContrastiveLoss(RepresentationLoss):
    """
    A basic contrastive loss that only does prediction/similarity comparison in one direction,
    only calculating a softmax of IJ similarity against all similarities with I.
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(AsymmetricContrastiveLoss, self).__init__(device, sample)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temp = temp

    def calculate_logits_and_labels(self, z_i, z_j):
        raise NotImplementedError

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets, _ = self.get_vector_forms(decoded_context_dist, target_dist, encoded_context_dist)

        z_i = decoded_contexts
        z_j = targets

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        logits, labels = self.calculate_logits_and_labels(z_i, z_j)
        return self.criterion(logits, labels)


class QueueAsymmetricContrastiveLoss(AsymmetricContrastiveLoss):
    """
    This implements algorithms that use a queue to maintain all the negative examples. The contrastive loss is
    calculated through the comparison of an image with its augmented version (positive example) and everything else
    in the queue (negative examples). The method used in MoCo.
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(QueueAsymmetricContrastiveLoss, self).__init__(device, sample)
        self.temp = temp

    def calculate_logits_and_labels(self, z_i, z_j):
        """
        z_i: dim (N, C). N - batch_size, C - representation dimension
        z_j: dim (N+K, C). K - number of entries in the queue
        """
        batch_size = z_i.shape[0]
        queue = z_j[batch_size:]
        z_j = z_j[:batch_size]

        l_pos = torch.einsum('nc,nc->n', [z_i, z_j]).unsqueeze(-1)  # Nx1
        l_neg = torch.einsum('nc,ck->nk', [z_i, queue.T])  # NxK

        logits = torch.cat([l_pos, l_neg], dim=1)  # Nx(1+K)
        logits /= self.temp

        # All l_pos are at the 0-th position
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

    def calculate_logits_and_labels(self, z_i, z_j):
        """
        z_i: dim (N, C). N - batch_size, C - representation dimension
        z_j: dim (N, C). The augmented images' representations
        """
        batch_size = z_i.shape[0]
        mask = torch.eye(batch_size)

        # Similarity of the original images with all other original images in current batch. Return a matrix of NxN.
        logits_aa = torch.matmul(z_i, z_i.T)  # NxN

        # The entry on the diagonal line is each image's similarity with itself, which can be a large number (e.g. for
        # normalized vectors, it is the max value 1). We want to exclude it in our calculation for cross entropy loss.
        logits_aa = logits_aa - mask

        # Similarity of original images and augmented images
        logits_ab = torch.matmul(z_i, z_j.T)  # NxN

        logits = torch.cat((logits_ab, logits_aa), 1)  # Nx2N
        label = torch.arange(batch_size, dtype=torch.long).to(self.device)
        return logits, label


class SymmetricContrastiveLoss(RepresentationLoss):
    """
    A contrastive loss that does prediction "in both directions," i.e. that calculates logits of IJ similarity against
    all similarities with J, and also all similarities with I, and calculates cross-entropy on both
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(SymmetricContrastiveLoss, self).__init__(device, sample)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temp = temp

    def calculate_similarity(self, z_i, z_j):
        raise NotImplementedError

    def calculate_mask(self, batch_size):
        raise NotImplementedError

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets, _ = self.get_vector_forms(decoded_context_dist, target_dist, encoded_context_dist)
        z_i = decoded_contexts
        z_j = targets
        batch_size = z_i.shape[0]

        mask = self.calculate_mask(batch_size)

        # Similarity of the original images with all other original images in current batch. Return a matrix of NxN.
        logits_aa = self.calculate_similarity(z_i, z_i)  # NxN

        # The entry on the diagonal line is each image's similarity with itself, which can be a large number (e.g. for
        # normalized vectors, it is the max value 1). We want to exclude it in our calculation for cross entropy loss.
        logits_aa = logits_aa - mask

        # Similarity of the augmented images with all other augmented images.
        logits_bb = self.calculate_similarity(z_j, z_j)  # NxN
        logits_bb = logits_bb - mask

        # Similarity of original images and augmented images
        logits_ab = self.calculate_similarity(z_i, z_j)  # NxN
        logits_ba = self.calculate_similarity(z_j, z_i)  # NxN

        # Each row now contains an image's similarity with the batch's augmented images & original images. This applies
        # to both original and augmented images (hence "symmetric").
        logits_i = torch.cat((logits_ab, logits_aa), 1)  # Nx2N
        logits_j = torch.cat((logits_ba, logits_bb), 1)  # Nx2N
        logits = torch.cat((logits_i, logits_j), axis=0)  # 2Nx2N

        label = torch.arange(batch_size, dtype=torch.long).to(self.device)
        labels = torch.cat((label, label), axis=0)

        return self.criterion(logits, labels)


class MatMulSymmetricContrastiveLoss(SymmetricContrastiveLoss):
    """
    A subclass of SymmetricContrastiveLoss that uses matrix multiplication as the similarity.
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(MatMulSymmetricContrastiveLoss, self).__init__(device, sample)
        self.temp = temp
        self.large_num = 1e9  # SimCLR's setting. TODO: Check if it's a reasonable number

    def calculate_similarity(self, z_i, z_j):
        return torch.matmul(z_i, z_j.T) / self.temp

    def calculate_mask(self, batch_size):
        return (torch.eye(batch_size) * self.large_num).to(self.device)


class CosineSymmetricContrastiveLoss(SymmetricContrastiveLoss):
    """
    A subclass of SymmetricContrastiveLoss that uses cosine similarity.
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(CosineSymmetricContrastiveLoss, self).__init__(device, sample)
        self.temp = temp

    def calculate_similarity(self, z_i, z_j):
        # Adjust dimensions so we can broadcast the tensors
        z_i = z_i[:, None, :]
        z_j = z_j[None, :, :]
        return F.cosine_similarity(z_i, z_j, dim=2) / self.temp

    def calculate_mask(self, batch_size):
        return torch.eye(batch_size).to(self.device)


class MSELoss(RepresentationLoss):
    def __init__(self, device, sample=False):
        super(MSELoss, self).__init__(device, sample)
        self.criterion = torch.nn.MSELoss()

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        decoded_contexts, targets, _ = self.get_vector_forms(decoded_context_dist, target_dist, encoded_context_dist)
        return self.criterion(decoded_contexts, targets)


class CEBLoss(RepresentationLoss):
    """
    A variational contrastive loss that implements information bottlenecking, but in a less conservative form
    than done by traditional VIB techniques
    """
    def __init__(self, device, beta=.1):
        super().__init__(device, sample=True)
        # TODO allow for beta functions
        self.beta = beta

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):

        z = decoded_context_dist.sample() # B x Z

        log_ezx = decoded_context_dist.log_prob(z) # B -> Log proba of each vector in z under the distribution it was sampled from
        log_bzy = target_dist.log_prob(z) # B -> Log proba of each vector in z under the distribution conditioned on its corresponding target

        cross_probas = torch.stack([target_dist.log_prob(z[i]) for i in range(z.shape[0])], dim=0) # BxB Log proba of each vector z under _all_ target distributions
        catgen = torch.distributions.Categorical(logits=cross_probas) # logits of shape BxB -> Batch categorical, one distribution per element in z over possible
                                                                      # targets/y values
        inds = torch.arange(start=0, end=len(z))
        i_yz = catgen.log_prob(inds) # The probability of the kth target under the kth Categorical distribution (probability of true y)
        loss = torch.mean(self.beta*(log_ezx - log_bzy) - i_yz)
        return loss


class MoCoAsymmetricContrastiveLoss(AsymmetricContrastiveLoss):
    """
    A contrastive loss that perform similarity comparison between IJ (original & augmented representations) and IQ
    (original & queued representations). This is used in MoCo.
    """
    def __init__(self, device, sample=False):
        super(MoCoAsymmetricContrastiveLoss, self).__init__(device, sample)

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets, _ = self.get_vector_forms(decoded_context_dist,
                                                             target_dist,
                                                             encoded_context_dist)

        z_i = decoded_contexts  # NxC
        batch_size = z_i.shape[0]
        z_j = targets[:batch_size]  # NxC
        queue = targets[batch_size:]  # KxC

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # zi and zj are both matrices of dim (n x c), since they are z vectors of dim c for every element in the batch
        # This einsum is constructing a vector of size n, where the nth element is a sum over C of the NCth elements
        # That is to say, the nth element is a dot product between the Nth C-dim vector of each matrix
        l_pos = torch.einsum('nc,nc->n', [z_i, z_j]).unsqueeze(-1)  # Nx1

        # (Cynthia) In Moco, for each image, it takes the sim with its augmented image (the line above), then compute
        # the image's similarity with everything else in the queue (the line below).
        l_neg = torch.einsum('nc,ck->nk', [z_i, queue.T])  # NxK

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temp

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        return self.criterion(logits, labels)
