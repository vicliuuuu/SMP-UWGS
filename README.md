# SMP-UWGS (Core Methods)

**SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction**<br>
ECCV 2026 · https://github.com/vicliuuuu/SMP-UWGS

本仓库仅发布论文中的**核心方法模块**（SMP 多分区 + DPR-Net + 混合物理损失），**不包含**完整训练、渲染与评测流水线。端到端训练与评估流程请参考 [SeaSplat](https://arxiv.org/abs/2503.10600) 及 3D Gaussian Splatting / VastGaussian 的工程组织方式，将下列模块接入其训练循环即可。

## 环境

与内部开发环境一致，使用 Conda 环境 `sea`：

```bash
conda activate sea
pip install -r requirements.txt
```

主要依赖：`torch==2.1.0+cu121`、`torchvision==0.16.0`、`kornia==0.7.3`、`numpy==1.26.0`、`plyfile`、`shapely`、`scipy` 等（见 `requirements.txt`）。

## 目录结构

```
SMP-UWGS/
├── dpr_net/                    # DPR-Net 可微物理渲染
│   ├── water_param_predictor.py   # WaterParamPredict (U-Net + SE-ResNet + Attention Gate)
│   ├── dbdr.py                    # DBDR: AttenuateNetV3 + BackscatterNetV2
│   ├── formation.py               # 水下图像形成与恢复 I = J·e^{-β_d z} + B_∞(1-e^{-β_b z})
│   ├── pa_dcp.py                  # Physics-Aware Dark Channel Prior
│   ├── losses.py                  # 混合物理-统计损失 (PA-DCP, BS/AT, α-bg 等)
│   └── depth_losses.py            # 边缘感知深度平滑
└── smp/                        # SMP-GAUSSIAN 多分区架构
    ├── data_partition.py          # 动态空间分区 + BARE 边界扩展 + 跨分区可见性融合
    ├── data_partition_depth.py    # DARWS 深度感知区域权重
    ├── seamless_merging.py        # 分区高斯无缝合并
    ├── appearance_network.py      # FiLM 外观解耦
    ├── graham_scan.py             # 分区可见性（凸包-图像交集）
    └── geom.py                    # 分区用基础几何类型 (BasicPointCloud, CameraInfo, storePly)
```

## 论文模块对照

| 论文 | 代码 |
|------|------|
| SMP-GAUSSIAN / BARE / 跨分区一致性 | `smp/data_partition.py`, `smp/seamless_merging.py` |
| DARWS | `smp/data_partition_depth.py` |
| WaterParamPredict | `dpr_net/water_param_predictor.py` |
| DBDR | `dpr_net/dbdr.py` |
| 水下图像形成方程 | `dpr_net/formation.py` |
| FiLM 外观解耦 | `smp/appearance_network.py` |
| PA-DCP 与混合损失 | `dpr_net/pa_dcp.py`, `dpr_net/losses.py`, `dpr_net/depth_losses.py` |

## 使用说明

本发布包提供可复用的 `nn.Module` 与分区算法，**不含** `train.py` / `render.py` / `metrics.py` 及示例 shell。集成时建议：

1. 在 3DGS 训练循环中渲染深度图，调用 `WaterParamPredictor` 初始化 `AttenuateNetV3` / `BackscatterNetV2`；
2. 用 `apply_underwater_formation` 合成水下图并计算 `dpr_net.losses` 中各项损失；
3. 大场景训练前调用 `ProgressiveDataPartitioning`（`data_partition_depth.py`）做分区与 DARWS 权重；
4. 分区训练结束后用 `seamless_merge` 合并高斯点云。

完整数据加载、多 GPU 调度、checkpoint 与指标计算请参照 SeaSplat / VastGaussian 仓库自行拼装。

## 未包含内容

- OTNN 等涉密数据集与专有预处理
- 完整训练 / 渲染 / 评测脚本
- 3DGS CUDA rasterizer 子模块（需从上游 3DGS 单独获取并编译）

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
