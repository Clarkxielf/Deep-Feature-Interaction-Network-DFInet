"""
Feature Extraction and Parameter Prediction networks
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from DFInet_2.src.models.pointnet_util import sample_and_group_multi

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}


class feat_extractor(nn.Module):
    def __init__(self, feature_dim=128):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)


        # Pointnet
        self.prepool0 = nn.Sequential(
            nn.Conv1d(2, feature_dim//8, 1),
            nn.GroupNorm(12, feature_dim//8),
            nn.ReLU(),
        )

        self.prepool1 = nn.Sequential(
            nn.Conv1d(feature_dim//8, feature_dim//8, 1),
            nn.GroupNorm(12, feature_dim//8),
            nn.ReLU(),
        )

        self.prepool2 = nn.Sequential(
            nn.Conv1d(feature_dim//8*2, feature_dim//8*4, 1),
            nn.GroupNorm(12, feature_dim//8*4),
            nn.ReLU(),
        )

        self.prepool3 = nn.Sequential(
            nn.Conv1d(feature_dim//8*4, feature_dim//8*4, 1),
            nn.GroupNorm(12, feature_dim//8*4),
            nn.ReLU(),
        )

        self.prepool4 = nn.Sequential(
            nn.Conv1d(feature_dim//8*8, feature_dim//8*8, 1),
            nn.GroupNorm(12, feature_dim//8*8),
            nn.ReLU(),
        )

        self.pooling = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, y):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """
        prepool_feat_x0 = self.prepool0(x.transpose(2, 1))
        prepool_feat_x1 = self.prepool1(prepool_feat_x0)

        prepool_feat_y0 = self.prepool0(y.transpose(2, 1))
        prepool_feat_y1 = self.prepool1(prepool_feat_y0)

        PFI_x1 = torch.cat([prepool_feat_x1, self.pooling(prepool_feat_y1).repeat(1, 1, prepool_feat_x1.shape[-1])], dim=-2)
        PFI_y1 = torch.cat([prepool_feat_y1, self.pooling(prepool_feat_x1).repeat(1, 1, prepool_feat_y1.shape[-1])], dim=-2)

        prepool_feat_x2 = self.prepool2(PFI_x1)
        prepool_feat_x3 = self.prepool3(prepool_feat_x2)

        prepool_feat_y2 = self.prepool2(PFI_y1)
        prepool_feat_y3 = self.prepool3(prepool_feat_y2)

        PFI_x3 = torch.cat([prepool_feat_x3, self.pooling(prepool_feat_y3).repeat(1, 1, prepool_feat_x3.shape[-1])], dim=-2)
        PFI_y3 = torch.cat([prepool_feat_y3, self.pooling(prepool_feat_x3).repeat(1, 1, prepool_feat_y3.shape[-1])], dim=-2)

        prepool_feat_x4 = self.prepool4(PFI_x3)
        prepool_feat_y4 = self.prepool4(PFI_y3)

        # prepool_feat_x = torch.cat([prepool_feat_x0, prepool_feat_x1, prepool_feat_x2, prepool_feat_x3, prepool_feat_x4], dim=-2)
        # prepool_feat_y = torch.cat([prepool_feat_y0, prepool_feat_y1, prepool_feat_y2, prepool_feat_y3, prepool_feat_y4], dim=-2)

        Fx_r = prepool_feat_x4.transpose(-1, -2)
        Fy_r = prepool_feat_y4.transpose(-1, -2)

        Fx_r = Fx_r / torch.norm(Fx_r, dim=-1, keepdim=True)
        Fy_r = Fy_r / torch.norm(Fy_r, dim=-1, keepdim=True)

        return Fx_r, Fy_r


class DGCNN(nn.Module):
    def __init__(self, feature_dim=128, num_neighbors=32):
        super(DGCNN, self).__init__()

        self.num_neighbors = num_neighbors

        self.conv1 = nn.Conv2d(2, feature_dim//2, kernel_size=1, bias=False)
        self.mlp1 = nn.Conv2d(self.num_neighbors, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(feature_dim//2, feature_dim, kernel_size=1, bias=False)
        self.mlp2 = nn.Conv2d(self.num_neighbors, 1, kernel_size=1)
        # self.conv3 = nn.Conv2d(8, 16, kernel_size=1, bias=False)
        # self.mlp3 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        # self.conv4 = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        # self.mlp4 = nn.Conv2d(num_neighbors, 1, kernel_size=1)
        self.conv5 = nn.Conv2d(feature_dim//2*3, feature_dim//2*3, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_dim//2)
        self.bn2 = nn.BatchNorm2d(feature_dim)
        # self.bn3 = nn.BatchNorm2d(16)
        # self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(feature_dim//2*3)


        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.prepool = nn.Sequential(
            nn.Conv1d(feature_dim//2*3*2, feature_dim, 1),
            nn.GroupNorm(12, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x, y):

        output_x0 = get_graph_feature(x.transpose(-1, -2), k=self.num_neighbors) #(B,D,N,K)

        output = F.relu(self.bn1(self.conv1(output_x0)))
        output_x1 = output.transpose(3, 1) #(B,K,N,D)
        output_x1 = self.mlp1(output_x1)  # (B,1,N,D)

        output = F.relu(self.bn2(self.conv2(output)))
        output_x2 = output.transpose(3, 1)
        output_x2 = self.mlp2(output_x2) #(B,1,N,D)

        output_x3 = torch.cat((output_x1, output_x2), dim=-1)
        output_x3 = output_x3.transpose(3, 1)#(B,D,N,1)
        # output_x3 = F.relu(self.bn5(self.conv5(output_x3))).squeeze()#(B,D,N)
        output_x3 = output_x3.squeeze()  #(B,D,N)


        output_y0 = get_graph_feature(y.transpose(-1, -2), k=self.num_neighbors) #(B,D,N,K)

        output = F.relu(self.bn1(self.conv1(output_y0)))
        output_y1 = output.transpose(3, 1) #(B,K,N,D)
        output_y1 = self.mlp1(output_y1)  # (B,1,N,D)

        output = F.relu(self.bn2(self.conv2(output)))
        output_y2 = output.transpose(3, 1)
        output_y2 = self.mlp2(output_y2) #(B,1,N,D)

        output_y3 = torch.cat((output_y1, output_y2), dim=-1)
        output_y3 = output_y3.transpose(3, 1) #(B,D,N,1)
        # output_y3 = F.relu(self.bn5(self.conv5(output_y3))).squeeze()#(B,D,N)
        output_y3 = output_y3.squeeze()


        PFI_x = torch.cat([output_x3, self.pooling(output_y3).repeat(1, 1, output_x3.shape[-1])], dim=-2)
        PFI_y = torch.cat([output_y3, self.pooling(output_x3).repeat(1, 1, output_y3.shape[-1])], dim=-2)

        prepool_feat_x = self.prepool(PFI_x)
        prepool_feat_y = self.prepool(PFI_y)

        Fx_r = prepool_feat_x.transpose(-1, -2)
        Fy_r = prepool_feat_y.transpose(-1, -2)

        Fx_r = Fx_r / torch.norm(Fx_r, dim=-1, keepdim=True)
        Fy_r = Fy_r / torch.norm(Fy_r, dim=-1, keepdim=True)

        return Fx_r, Fy_r



def get_graph_feature(x, k=20):

    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    feature = feature.permute(0, 3, 1, 2)

    return feature


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -(xx + inner + xx.transpose(2, 1).contiguous())

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class G_feat_ParameterPredictionNet(nn.Module):
    def __init__(self):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2),
        )

    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)
        # concatenated = torch.cat([x[0], x[1]], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha

class L_feat_ParameterPredictionNet(nn.Module):
    def __init__(self):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2),
        )

    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], (0, 1), mode='constant', value=0)
        ref_padded = F.pad(x[1], (0, 1), mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)
        # concatenated = torch.cat([x[0], x[1]], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class weighted_mlp(nn.Module):
    def __init__(self, feature_dim=32):
        super(weighted_mlp, self).__init__()

        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(2, feature_dim//2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(feature_dim//2, 2, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(feature_dim//2)
        self.bn2 = nn.BatchNorm2d(2)


    def forward(self, input):

        input = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(input)))

        return output
