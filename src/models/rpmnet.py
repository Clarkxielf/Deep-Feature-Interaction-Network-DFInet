import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

from DFInet_2.src.common.torch import to_numpy, dict_all_to_device
from DFInet_2.src.models.pointnet_util import square_distance, angle_difference
from DFInet_2.src.models.feature_nets import feat_extractor, DGCNN, G_feat_ParameterPredictionNet, L_feat_ParameterPredictionNet, weighted_mlp
from DFInet_2.src.common.math_torch import se3

_logger = logging.getLogger(__name__)

_EPS = 1e-5  # To prevent division by zero

def match_matrix(src, ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert src.shape[-1] == ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(src, ref)
    elif metric == 'angle':
        feat_src_norm = src / (torch.norm(src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = ref / (torch.norm(ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix

def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and columnd
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha

def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 1] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


class FINet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.add_slack = not args.no_slack
        self.num_sk_iter = args.num_sk_iter

    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def forward(self, data, num_iter: int = 1):
        """Forward pass for RPMNet

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, 6)
                    'points_ref': Reference points (B, K, 6)
            num_iter (int): Number of iterations. Recommended to be 2 for training

        Returns:
            transform: Transform to apply to source points such that they align to reference
            src_transformed: Transformed source points
        """
        endpoints = {}

        xyz_ref = data['points_ref'][:, :, :2]
        xyz_src = data['points_src'][:, :, :2]
        xyz_src_t = xyz_src


        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []

        for i in range(num_iter):

            G_Fx, G_Fy = self.feat_extractor(xyz_src_t, xyz_ref)

            G_feat_distance = match_matrix(G_Fx, G_Fy)
            G_feat_beta, G_feat_alpha = self.G_feat_weights_net([xyz_src_t, xyz_ref])
            G_feat_affinity = self.compute_affinity(G_feat_beta, G_feat_distance, alpha=G_feat_alpha)
            G_feat_log_perm_matrix = sinkhorn(G_feat_affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            G_feat_perm_matrix = torch.exp(G_feat_log_perm_matrix)

            L_Fx, L_Fy = self.DGCNN(xyz_src_t, xyz_ref)

            L_feat_distance = match_matrix(L_Fx, L_Fy)
            L_feat_beta, L_feat_alpha = self.L_feat_weights_net([xyz_src_t, xyz_ref])
            L_feat_affinity = self.compute_affinity(L_feat_beta, L_feat_distance, alpha=L_feat_alpha)
            L_feat_log_perm_matrix = sinkhorn(L_feat_affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            L_feat_perm_matrix = torch.exp(L_feat_log_perm_matrix)

            feat_perm_matrix = torch.cat([G_feat_perm_matrix.unsqueeze(1), L_feat_perm_matrix.unsqueeze(1)], 1)
            # feat_perm_matrix = self.weighted_mlp(feat_perm_matrix)

            weighted_G_feat_perm_matrix = feat_perm_matrix[:, 0, :, :].topk(k=1, dim=-1)[0]
            weighted_L_feat_perm_matrix = feat_perm_matrix[:, 1, :, :].topk(k=1, dim=-1)[0]
            weighted_perm_matrix = torch.cat([weighted_G_feat_perm_matrix, weighted_L_feat_perm_matrix], -1)
            weighted_perm_matrix = torch.softmax(weighted_perm_matrix, -1)
            perm_matrix = weighted_perm_matrix[..., 0][..., None]*G_feat_perm_matrix\
                          + weighted_perm_matrix[..., 1][..., None]*L_feat_perm_matrix


            weighted_ref = perm_matrix @ xyz_ref / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

            # Compute transform and transform points
            transform = compute_rigid_transform(xyz_src, weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            xyz_src_t = se3.transform(transform.detach(), xyz_src)

            transforms.append(transform)
            # all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(perm_matrix)
            # all_weighted_ref.append(weighted_ref)
            # all_beta.append(to_numpy(feat_beta))
            # all_alpha.append(to_numpy(feat_alpha))


        # endpoints['perm_matrices_init'] = all_gamma
        endpoints['perm_matrices'] = all_perm_matrices
        # endpoints['weighted_ref'] = all_weighted_ref
        # endpoints['beta'] = np.stack(all_beta, axis=0)
        # endpoints['alpha'] = np.stack(all_alpha, axis=0)

        return transforms, endpoints



class FINetEarlyFusion(FINet):
    """Early fusion implementation of RPMNet, as described in the paper"""
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        self.feat_extractor = feat_extractor(feature_dim=args.feat_dim)
        self.DGCNN = DGCNN(feature_dim=args.feat_dim, num_neighbors=args.num_neighbors)


        self.G_feat_weights_net = G_feat_ParameterPredictionNet()
        self.L_feat_weights_net = L_feat_ParameterPredictionNet()

        self.weighted_mlp = weighted_mlp(feature_dim=args.feat_dim)



def get_model(args: argparse.Namespace) -> FINet:
    return FINetEarlyFusion(args)
