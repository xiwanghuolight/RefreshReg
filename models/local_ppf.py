import torch
import torch.nn as nn

from utils import sample_and_group, angle


class LocalFeatureFused(nn.Module):
    def __init__(self, in_dim, out_dims):
        super(LocalFeatureFused, self).__init__()
        self.blocks = nn.Sequential()
        for i, out_dim in enumerate(out_dims):
            self.blocks.add_module(f'conv2d_{i}',
                                   nn.Conv2d(in_dim, out_dim, 1, bias=False))
            self.blocks.add_module(f'in_{i}',
                                   nn.InstanceNorm2d(out_dims))
            self.blocks.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        '''
        :param x: (B, C1, K, M)
        :return: (B, C2, M)
        '''
        x = self.blocks(x)
        x = torch.max(x, dim=2)[0]
        return x


class PointPair(nn.Module):
    def __init__(self, feats_dim, k, radius):
        super().__init__()
        self.k = k
        self.radius = radius
        self.local_feature_fused = LocalFeatureFused(in_dim=10,
                                                     out_dims=feats_dim)

    def forward(self, coords, feats):
        '''

        :param coors: (B, 3, N)
        :param feats: (B, 3, N)
        :param k: int
        :return: (B, C, N)
        '''

        feats = feats.permute(0, 2, 1).contiguous()
        coords = coords.permute(0, 2, 1).contiguous()
        new_xyz, new_points, grouped_inds, grouped_xyz = \
            sample_and_group(xyz=coords,
                             points=feats,
                             M=-1,
                             radius=self.radius,
                             K=self.k)
        nr_d = angle(feats[:, :, None, :], grouped_xyz)
        ni_d = angle(new_points[..., 3:], grouped_xyz)
        nr_ni = angle(feats[:, :, None, :], new_points[..., 3:])
        d_norm = torch.norm(grouped_xyz, dim=-1)
        ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, N, K, 4)
        new_points = torch.cat([new_points[..., :3], ppf_feat], dim=-1)

        coords = torch.unsqueeze(coords, dim=2).repeat(1, 1, min(self.k, new_points.size(2)), 1)
        new_points = torch.cat([coords, new_points], dim=-1)
        feature_local = new_points.permute(0, 3, 2, 1).contiguous()  # (B, C1 + 3, K, M)
        feature_local = self.local_feature_fused(feature_local)
        return feature_local


class GCE(nn.Module):
    def __init__(self, feats_dim, gcn_k, ppf_k, radius, bottleneck):
        super().__init__()
        if bottleneck:
            self.ppf = PointPair([feats_dim // 2, feats_dim, feats_dim // 2], ppf_k, radius)
            self.fused = nn.Sequential(
                nn.Conv1d(feats_dim + feats_dim // 2, feats_dim + feats_dim // 2, 1),
                nn.InstanceNorm1d(feats_dim + feats_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(feats_dim + feats_dim // 2, feats_dim, 1),
                nn.InstanceNorm1d(feats_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.ppf = PointPair([feats_dim, feats_dim * 2, feats_dim], ppf_k, radius)

    def forward(self, coords, feats, normals):
        feats_ppf = self.ppf(coords, normals)
        return feats_ppf


class LocalInteractive(nn.Module):
    def __init__(self, feat_dims, gcn_k, ppf_k, radius, bottleneck, nhead):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(GCE(feat_dims, gcn_k, ppf_k, radius, bottleneck))

    def forward(self, coords1, feats1, coords2, feats2, normals1, normals2):
        '''

        :param coords1: (B, 3, N)
        :param feats1: (B, C, N)
        :param coords2: (B, 3, M)
        :param feats2: (B, C, M)
        :return: feats1=(B, C, N), feats2=(B, C, M)
        '''
        for block in self.blocks:
            feats1 = block(coords1, feats1, normals1)
            feats2 = block(coords2, feats2, normals2)

        return feats1, feats2