# RefreshReg: Receptive Field Reshaping and Multi-Layer Consistency Filtering for Point Cloud Registration



## 🌟 Introduction

Point cloud registration is a crucial task in the field of 3D processing research, which aims to align two or more point cloud scans into the same coordinate system. A significant factor limiting the performance of point cloud registration is the low proportion of inlier correspondences between two unaligned point clouds. It is particularly pronounced when the overlap between two scenes is low. Based on this observation, we propose a novel point cloud registration framework that enhances the proportion of correct correspondences via two aspects: extracting richer global geometric information for accurate identification of overlapping regions, and rejecting outliers based on spatial feature consistency. During the feature extraction phase, we first encode local geometry utilizing the Point Pair Features and then propose the Dual Graph Convolution module to reshape the receptive field, thereby expanding perception beyond small local areas. In the transformation estimation phase, we design a filtering module based on a multi-layer decoder. We extract point cloud features at different resolutions and select high-confidence point cloud pairs for registration based on the consistency of correspondences. We test the performance of our method on four datasets (3DMatch, ScanNet, KITTI, and MVP-RG). Compared with state-of-the-art approach NMCT, our method achieves improvements of 6% / 38% on KITTI / MVP-RG. Additionally, our filtering approach enhances the operational speed of RANSAC by more than 300%.

## 🚀 Contributions

- **local-to-global framework**: We present RefreshReg, a point cloud registration framework featuring a local-to-global strategy. First, local descriptors capture fine-grained geometric information. Second, the receptive field is reshaped via DGC. Finally, we utilize a bilinear response module to derive global geometric features
- **Dual Graph Convolution module**: We propose a Dual Graph Convolution (DGC) module to reshape the receptive field, integrating spatial geometric information and feature context, which contributes to the accurate inference of overlapping regions in point clouds.
- **filtering strategy**: We propose a filtering module based on a multi-layer decoder. It adaptively fuses multi-resolutional features and rejects outliers, which helps to increase the proportion of correct correspondences during matching.

## 📊 Performance

| Benchmark   | RR                  | State-of-the-Art | RefreshReg | Improvement |
|----------|---------------------|------------------|------------|-------------|
| 3DMatch | Registration Recall | 71.7% (NMCT)     | 73.5%      | 1.8%        |
| 3DLoMatch | Registration Recall | 71.7% (NMCT)     | 73.5%      | 1.8%        |
| KITTI    | Registration Recall | 99.8% (NMCT)     | 99.8%      | 6%*         |
| MVP-RG   | L_RMSE              | 0.115 (NMCT)     | 0.097      | 38%         |

*Improvement in other key metrics (e.g., RTE/RRE).

For detailed results, refer to the [paper](https://arxiv.org/abs/xxxx.xxxx).

## 🛠️ Installation

### Prerequisites
- Python 3.8
- PyTorch 1.10
- CUDA 11.3
- Other dependencies: `numpy`, `open3d`, `torch-scatter`, `einops`

### Install
```bash
git clone https://github.com/xiwanghuolight/RefreshReg.git
cd RefreshReg
pip install -r requirements.txt
```

## 📚 Usage

### Datasets
Download the datasets used in the paper:
- [3DMatch/3DLoMatch](https://3dmatch.cs.princeton.edu/)
- [ScanNet](https://www.scan-net.org/)
- [KITTI](http://www.cvlibs.net/datasets/kitti/)
- [MVP-RG](https://github.com/paul007pl/MVP_RG)

## 🎓 Citation
