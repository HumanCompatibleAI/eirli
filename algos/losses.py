from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
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