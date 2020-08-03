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
    only calculating a softmax of IJ similarity against all similarities with I. Represents InfoNCE
    used in original CPC paper
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(AsymmetricContrastiveLoss, self).__init__(device, sample)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temp = temp

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets, _ = self.get_vector_forms(decoded_context_dist, target_dist, encoded_context_dist)

        z_i = decoded_contexts
        z_j = targets

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        num_contexts = z_i.shape[0]
        num_targets = z_j.shape[0]

        if num_targets > num_contexts:
            z_all = torch.cat([z_i, z_j], 0)
            z_j = z_j[:num_contexts]
        else:
            z_all = torch.cat((z_i, z_j), 0)

        # zi and zj are both matrices of dim (n x c), since they are z vectors of dim c for every element in the batch
        # This einsum is constructing a vector of size n, where the nth element is a sum over C of the NCth elements
        # That is to say, the nth element is a dot product between the Nth C-dim vector of each matrix
        sim_ij = torch.einsum('nc,nc->n', [z_i, z_j]).unsqueeze(-1)  # Nx1

        # TODO the num_targets in the below expression used to be batch size. Changed it to num_targets but this might be wrong
        # (don't 100% understand the logic of the normalization term here)
        sim_ik = (torch.einsum('nc,mc->n', [z_i, z_all]).unsqueeze(-1) - 1) / (2 * num_targets - 1)

        logits = torch.cat((sim_ij, sim_ik), 1)
        logits /= self.temp

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        return self.criterion(logits, labels)


class SymmetricContrastiveLoss(RepresentationLoss):
    """
    A contrastive loss that does prediction "in both directions," i.e. that calculates logits of IJ similarity against
    all similarities with J, and also all similarities with I, and calculates cross-entropy on both
    """
    def __init__(self, device, sample=False, temp=0.1):
        super(SymmetricContrastiveLoss, self).__init__(device, sample)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.temp = temp

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets, _ = self.get_vector_forms(decoded_context_dist, target_dist, encoded_context_dist)
        z_i = decoded_contexts
        z_j = targets

        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        z_all = torch.cat((z_i, z_j), 0)  # 2NxC

        # zi and zj are both matrices of dim (n x c), since they are z vectors of dim c for every element in the batch
        # This einsum is constructing a vector of size n, where the nth element is a sum over C of the NCth elements
        # That is to say, the nth element is a dot product between the Nth C-dim vector of each matrix
        sim_ij = torch.einsum('nc,nc->n', [z_i, z_j]).unsqueeze(-1)  # Nx1

        sim_ik = (torch.einsum('nc,mc->n', [z_i, z_all]).unsqueeze(-1) - 1) / (2 * batch_size - 1)  # Nx1
        sim_jk = (torch.einsum('nc,mc->n', [z_j, z_all]).unsqueeze(-1) - 1) / (2 * batch_size - 1)  # Nx1

        logit_i = torch.cat((sim_ij, sim_ik), 1)  # Nx2
        logit_j = torch.cat((sim_ij, sim_jk), 1)  # Nx2
        logits = torch.cat((logit_i, logit_j), 0)  # 2Nx2

        logits /= self.temp

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # 2Nx1
        return self.criterion(logits, labels)


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


class SimCLRSymmetricContrastiveLoss(SymmetricContrastiveLoss):
    """
    A contrastive loss that does prediction "in both directions," i.e. that calculates logits of IJ similarity against
    all similarities with J, and also all similarities with I, and calculates cross-entropy on both. Adopted from
    SimCLR's implementation.
    """
    def __init__(self, device, sample=False):
        super(SimCLRSymmetricContrastiveLoss, self).__init__(device, sample)

    def __call__(self, decoded_context_dist, target_dist, encoded_context_dist=None):
        # decoded_context -> representation of context + optional projection head
        # target -> representation of target + optional projection head
        # encoded_context -> not used by this loss
        decoded_contexts, targets, _ = self.get_vector_forms(decoded_context_dist, target_dist, encoded_context_dist)
        z_i = decoded_contexts
        z_j = targets

        batch_size = z_i.shape[0]

        mask = torch.eye(batch_size).to(self.device)

        # In the updated official implementation, matrix multiplication (vs. cosine similarity in the paper) is used.
        logits_aa = torch.matmul(z_i, z_i.T) / self.temp
        # The entry on the diagonal line is each image's similarity with itself, which is always 1 (the max value)
        # for normalized vectors. We do not want to include this in our logits.
        logits_aa = logits_aa - mask
        logits_bb = torch.matmul(z_j, z_j.T) / self.temp
        logits_bb = logits_bb - mask
        logits_ab = torch.matmul(z_i, z_j.T) / self.temp
        logits_ba = torch.matmul(z_j, z_i.T) / self.temp

        logits_i = torch.cat((logits_ab, logits_aa), 1)
        logits_j = torch.cat((logits_ba, logits_bb), 1)

        label = torch.arange(batch_size, dtype=torch.long).to(self.device)
        logits = torch.cat((logits_i, logits_j), axis=0)
        labels = torch.cat((label, label), axis=0)
        return self.criterion(logits, labels)


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
