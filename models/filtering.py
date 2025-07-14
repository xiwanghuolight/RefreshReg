import torch
from utils import to_tensor


def get_coor_points(source_feats_npy, target_feats_npy, target_npy, use_cuda):
    dists = torch.cdist(to_tensor(source_feats_npy, use_cuda), to_tensor(target_feats_npy, use_cuda)) # (n, m)
    inds = torch.min(dists, dim=-1)[1].cpu().numpy()
    return inds


def filtering(source_npy, target_npy, source_feats, target_feats, voxel_size, use_cuda=True):
    source_feats_h, source_feats_m, source_feats_l = source_feats
    target_feats_h, target_feats_m, target_feats_l = target_feats

    coor_inds_h = get_coor_points(source_feats_h, target_feats_h, target_npy, use_cuda)
    coor_inds_m = get_coor_points(source_feats_m, target_feats_m, target_npy, use_cuda)
    coor_inds_l = get_coor_points(source_feats_l, target_feats_l, target_npy, use_cuda)

    coor_new = []
    # Iterate through the lists and check the condition
    for i in range(len(coor_inds_h)):
        # Count the number of lists where the value at the current index is the same as in coor_inds_h
        count = sum([coor_inds_h[i] == coor_inds_m[i], coor_inds_h[i] == coor_inds_l[i], coor_inds_m[i] == coor_inds_l[i]])
        coor_new.append(count >= 1)

    source_npy = source_npy[coor_new]
    source_feats_h = source_feats_h[coor_new]

    filtered =  [source_npy, target_npy, source_feats_h, target_feats_h]
    return filtered
  