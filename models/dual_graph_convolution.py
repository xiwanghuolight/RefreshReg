import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_neighbors(x, feature, k=20, idx=None):
    '''
        input: x, [B,3,N]
               feature, [B,C,N]
        output: neighbor_x, [B,6,N,K]
                neighbor_feat, [B,2C,N,k]
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,1).contiguous()  # batch_size * num_points * k + range(0, batch_size*num_points)
    neighbor_x = x.view(batch_size * num_points, -1)[idx, :]
    neighbor_x = neighbor_x.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    neighbor_x = torch.cat((neighbor_x - x, x), dim=3).permute(0, 3, 1, 2)

    _, num_dims, _ = feature.size()

    feature = feature.transpose(2,1).contiguous()
    neighbor_feat = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_feat = neighbor_feat.view(batch_size, num_points, k, num_dims)
    feature = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    neighbor_feat = torch.cat((neighbor_feat - feature, feature), dim=3).permute(0, 3, 1, 2)

    return neighbor_x, neighbor_feat



class DGC(nn.Module):
    def __init__(self, input_features_dim):
        super(DGC, self).__init__()

        self.conv_mlp1 = nn.Conv2d(6, input_features_dim // 2, 1)
        self.bn_mlp1 = nn.BatchNorm2d(input_features_dim // 2)

        self.conv_mlp2 = nn.Conv2d(input_features_dim * 2, input_features_dim // 2, 1)
        self.bn_mlp2 = nn.BatchNorm2d(input_features_dim // 2)


    def forward(self, xyz, features, k):

        neighbor_xyz, neighbor_feat = get_neighbors(xyz, features, k=k)

        # geometric graph
        neighbor_xyz = F.relu(self.bn_mlp1(self.conv_mlp1(neighbor_xyz)))  # B,C/2,N,k

        # feature graph
        neighbor_feat = F.relu(self.bn_mlp2(self.conv_mlp2(neighbor_feat)))  # B,C/2,N,k

        graph_encoding = torch.cat((neighbor_xyz, neighbor_feat), dim=1)  # B,C,N,k
        graph_encoding = graph_encoding.max(dim=-1, keepdim=False)[0]  # B,C,N

        return graph_encoding
