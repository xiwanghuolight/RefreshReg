# RefreshReg: Receptive Field Reshaping and Multi-Layer Consistency Filtering for Point Cloud Registration



## 🌟 Introduction

Point cloud registration is a crucial task in the field of 3D processing research, which aims to align two or more point cloud scans into the same coordinate system. A significant factor limiting the performance of point cloud registration is the low proportion of inlier correspondences between two unaligned point clouds. It is particularly pronounced when the overlap between two scenes is low. Based on this observation, we propose a novel point cloud registration framework that enhances the proportion of correct correspondences via two aspects: extracting richer global geometric information for accurate identification of overlapping regions, and rejecting outliers based on spatial feature consistency. During the feature extraction phase, we first encode local geometry utilizing the Point Pair Features and then propose the Dual Graph Convolution module to reshape the receptive field, thereby expanding perception beyond small local areas. In the transformation estimation phase, we design a filtering module based on a multi-layer decoder. We extract point cloud features at different resolutions and select high-confidence point cloud pairs for registration based on the consistency of correspondences. We test the performance of our method on four datasets (3DMatch, ScanNet, KITTI, and MVP-RG). Compared with state-of-the-art approach NMCT, our method achieves improvements of 6% / 38% on KITTI / MVP-RG. Additionally, our filtering approach enhances the operational speed of RANSAC by more than 300%.

## 🚀 Contributions

- **local-to-global framework**: We present RefreshReg, a point cloud registration framework featuring a local-to-global strategy. First, local descriptors capture fine-grained geometric information. Second, the receptive field is reshaped via DGC. Finally, we utilize a bilinear response module to derive global geometric features
- **Dual Graph Convolution module**: We propose a Dual Graph Convolution (DGC) module to reshape the receptive field, integrating spatial geometric information and feature context, which contributes to the accurate inference of overlapping regions in point clouds.
- **filtering strategy**: We propose a filtering module based on a multi-layer decoder. It adaptively fuses multi-resolutional features and rejects outliers, which helps to increase the proportion of correct correspondences during matching.

## 📊 Performance

### 3DMatch
| Benchmark |   RR(%)  |  FMR(%)  |   IR(%)  |
|---------- |-------|-------|-------|
| 3DMatch   | 94.2 | 98.3 | 66.9 |
| 3DLoMatch | 73.5 | 85.2 | 39.1 |

### Kitti odometry
| Benchmark | RRE(°) | RTE(cm)  |  RR(%)  |
|---------- |-----|------|------|
|   Kitti   | 5.9 | 0.23 | 99.8|

### MVP-RG
| Benchmark | RRE(°) | RTE(cm)  |  RR(%)  |
|---------- |-----|------|------|
|   MVP-RG   | 7.36 | 0.047 | 99.8|


## 🛠️ Installation

### Prerequisites
- Python 3.8
- PyTorch 1.10
- CUDA 11.3
- Other dependencies: `numpy`, `open3d`, `torch-scatter`, `einops`

### Install
```bash
conda create -n RefreshReg python==3.8
conda activate RefreshReg
git clone https://github.com/xiwanghuolight/RefreshReg.git
cd RefreshReg
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
pip install -r requirements.txt
```

## 📚 Usage

### Datasets
Download the datasets used in the paper:
- [3DMatch/3DLoMatch](https://github.com/prs-eth/OverlapPredator))
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
- [MVP-RG](https://mvp-dataset.github.io/MVP/Registration.html)

## 🎓 Citation
