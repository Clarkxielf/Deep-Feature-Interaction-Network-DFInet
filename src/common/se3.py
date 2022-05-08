""" 2-d rigid body transformation group"""


import torch


def transform(T, src):
    """ Applies the SE3 transform

    Args:
        T: SE3 transformation matrix of size (B, 2, 3)
        src: Points to be transformed (B, N, 2)

    Returns:
        transformed points of size (B, N, 2)

    """

    if len(src.shape)==3:
        rot_mat = T[:, :, :2]  # (B, 2, 2)
        translation = T[:, :, 2]  # (B, 2)

        src_transformed = torch.matmul(src, rot_mat.transpose(-1, -2)) + translation[:, None, :]
    elif len(src.shape)==2:
        rot_mat = T[:, :2]  # (2, 2)
        translation = T[:, 2]  # (2)

        src_transformed = torch.matmul(src, rot_mat.transpose(-1, -2)) + translation[None, :]

    return src_transformed
