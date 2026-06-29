# SMP-UWGS

Official repository for:

**SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction**<br>
Accepted by ECCV 2026.

- Project page: https://vicliuuuu.github.io/SMP-UWGS/
- GitHub: https://github.com/vicliuuuu/SMP-UWGS

本仓库已发布论文中的**核心方法模块**（SMP 多分区 + DPR-Net + 混合物理损失）。

## Overview

非常荣幸地宣布，我的论文 **SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction** 已被 ECCV 2026 接收。

该工作聚焦于大规模水下三维重建任务。水下场景中普遍存在光衰减、后向散射、浑浊介质以及深度相关颜色失真等问题，使得现有三维重建方法在复杂水下环境中难以同时兼顾重建质量、颜色一致性和训练效率。

为此，我们提出了 SMP-UWGS，一种耦合物理建模与几何优化的水下 3DGS 框架。该方法通过多分区 Gaussian 表示实现大规模场景的高效区域优化，并引入可微物理渲染模块，对水下光学参数进行估计与联合优化，从而更好地建模深度相关的衰减和散射效应。

Underwater 3D reconstruction struggles with light attenuation, backscattering, turbidity, and depth-dependent color distortion. SMP-UWGS couples physics modeling with geometric optimization, using multi-partition Gaussian representations for scalable optimization and a differentiable physics-based rendering module to jointly estimate optical parameters.

## Release Scope

This release provides reusable `nn.Module` implementations and partitioning algorithms. It does **not** include complete training, rendering, evaluation scripts, private datasets, or pretrained checkpoints.

端到端训练与评估流程请参考 [SeaSplat](https://arxiv.org/abs/2503.10600) 及 3D Gaussian Splatting / VastGaussian 的工程组织方式，将下列模块接入其训练循环即可。

## Environment

与内部开发环境一致，使用 Conda 环境 `sea`：

```bash
conda activate sea
pip install -r requirements.txt
```

主要依赖：`torch==2.1.0+cu121`、`torchvision==0.16.0`、`kornia==0.7.3`、`numpy==1.26.0`、`plyfile`、`shapely`、`scipy` 等（见 `requirements.txt`）。

## Repository Structure

```text
SMP-UWGS/
├── dpr_net/                    # DPR-Net 可微物理渲染
│   ├── water_param_predictor.py   # WaterParamPredict (U-Net + SE-ResNet + Attention Gate)
│   ├── dbdr.py                    # DBDR: AttenuateNetV3 + BackscatterNetV2
│   ├── formation.py               # 水下图像形成与恢复 I = J·e^{-β_d z} + B_∞(1-e^{-β_b z})
│   ├── pa_dcp.py                  # Physics-Aware Dark Channel Prior
│   ├── losses.py                  # 混合物理-统计损失 (PA-DCP, BS/AT, α-bg 等)
│   └── depth_losses.py            # 边缘感知深度平滑
├── smp/                        # SMP-GAUSSIAN 多分区架构
│   ├── data_partition.py          # 动态空间分区 + BARE 边界扩展 + 跨分区可见性融合
│   ├── data_partition_depth.py    # DARWS 深度感知区域权重
│   ├── seamless_merging.py        # 分区高斯无缝合并
│   ├── appearance_network.py      # FiLM 外观解耦
│   ├── graham_scan.py             # 分区可见性（凸包-图像交集）
│   └── geom.py                    # 分区用基础几何类型 (BasicPointCloud, CameraInfo, storePly)
├── docs/                       # Project page
├── requirements.txt
└── README.md
```

## Module Mapping

| Paper component | Code |
| --- | --- |
| SMP-GAUSSIAN / BARE / cross-partition consistency | `smp/data_partition.py`, `smp/seamless_merging.py` |
| DARWS | `smp/data_partition_depth.py` |
| WaterParamPredict | `dpr_net/water_param_predictor.py` |
| DBDR | `dpr_net/dbdr.py` |
| Underwater image formation | `dpr_net/formation.py` |
| FiLM appearance decoupling | `smp/appearance_network.py` |
| PA-DCP and hybrid physics losses | `dpr_net/pa_dcp.py`, `dpr_net/losses.py`, `dpr_net/depth_losses.py` |

## Usage Notes

集成时建议：

1. 在 3DGS 训练循环中渲染深度图，调用 `WaterParamPredictor` 初始化 `AttenuateNetV3` / `BackscatterNetV2`；
2. 用 `apply_underwater_formation` 合成水下图并计算 `dpr_net.losses` 中各项损失；
3. 大场景训练前调用 `ProgressiveDataPartitioning`（`data_partition_depth.py`）做分区与 DARWS 权重；
4. 分区训练结束后用 `seamless_merge` 合并高斯点云。

完整数据加载、多 GPU 调度、checkpoint 与指标计算请参照 SeaSplat / VastGaussian 仓库自行拼装。

## Not Included

- OTNN 等涉密数据集与专有预处理
- 完整训练 / 渲染 / 评测脚本
- Pretrained checkpoints
- 3DGS CUDA rasterizer 子模块（需从上游 3DGS 单独获取并编译）

## Acknowledgements

我需要诚挚感谢倪国威师兄在研究构思与方向上的建议，感谢郭修睿在论文写作过程中的帮助，感谢晨洋师兄在实验经费相关事宜中提供的支持，感谢曹城玮与吴志伟师兄在论文修改和 rebuttal 阶段提出的宝贵建议。

最后，感谢 Willand 的伙伴们给予我的信任、包容与指导，感谢刘帅和薛凌在实验经费方面提供的支持，感谢君君在长期研究与写作过程中给予的陪伴、理解与支持。

We also thank Guowei Ni for guidance on the research direction, Xiurui Guo for help with writing, Chengwei Cao and Zhiwei Wu for valuable feedback during revision and rebuttal, our partners at Willand for trust and guidance, Shuai Liu and Ling Xue for funding support, and Junjun for long-term support throughout this work.

## Contact

欢迎各位同学、老师批评指正。如有相关研究交流或合作意向，可在项目中留言或通过邮箱联系：yuxuanliu0128@163.com。来信请提前注明来意。

For research discussions or collaborations, please leave a comment in the project or contact: yuxuanliu0128@163.com.

## Citation

```bibtex
@inproceedings{liu2026smpuwgs,
  title={SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction},
  author={Liu, Yuxuan and Zhang, Jinhui},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## License

见 `LICENSE` 及上游 3DGS / VastGaussian 许可条款。
