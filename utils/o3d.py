import copy
import numpy as np
import torch
import open3d as o3d
import random


def npy2pcd(npy):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    return pcd


def pcd2npy(pcd):
    npy = np.array(pcd.points)
    return npy

def npy2feat(npy):
    feats = o3d.registration.Feature()
    feats.data = npy.T
    return feats


def normal(pcd, radius=0.1, max_nn=30, loc=(0, 0, 0)):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn),
                         fast_normal_computation=False)
    pcd.orient_normals_towards_camera_location(loc)
    return pcd


# the speed is slow here ? we should test it later.
def get_correspondences(src_ply, tgt_ply, transf, search_radius, K=None):
    src_ply = copy.deepcopy(src_ply)
    src_ply.transform(transf)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_ply)
    src_npy = pcd2npy(src_ply)
    corrs = []
    for i in range(src_npy.shape[0]):
        point = src_npy[i]
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, search_radius)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            corrs.append([i, j])
    return np.array(corrs)


def voxel_ds(ply, voxel_size):
    return ply.voxel_down_sample(voxel_size=voxel_size)



def execute_global_registration(source, target, source_feats, target_feats, voxel_size):
    distance_threshold = voxel_size
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feats, target_feats, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    transformation = result.transformation
    estimate = copy.deepcopy(source)
    estimate.transform(transformation)
    return transformation, estimate
