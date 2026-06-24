# SMP-UWGS

  Official project page for:

  **SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction**

  Accepted by ECCV 2026.

  Code, data, pretrained models, and project materials will be released soon.

  Project page: https://vicliuuuu.github.io/SMP-UWGS/

非常荣幸地宣布，我的论文SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction已被ECCV 2026接收。

该工作聚焦于大规模水下三维重建任务。水下场景中普遍存在光衰减、后向散射、浑浊介质以及深度相关颜色失真等问题，使得现有三维重建方法在复杂水下环境中难以同时兼顾重建质量、颜色一致性和训练效率。

为此，我提出了SMP-UWGS，一种耦合物理建模与几何优化的水下3DGS框架。该方法通过多分区Gaussian表示实现大规模场景的高效区域优化，并引入可微物理渲染模块，对水下光学参数进行估计与联合优化，从而更好地建模深度相关的衰减和散射效应。

最终论文、代码及更多项目材料将陆续整理并发布，敬请关注。

我需要诚挚感谢倪国威师兄在研究构思与方向上的建议，感谢郭修睿在论文写作过程中的帮助，感谢曹城玮与吴志伟师兄在论文修改和 rebuttal 阶段提出的宝贵建议。

最后，感谢 Willand 的伙伴们给予我的信任、包容与指导，感谢刘帅和薛凌在实验经费方面提供的支持，感谢君君在长期研究与写作过程中给予的陪伴、理解与支持。

欢迎各位同学、老师批评指正。如有相关研究交流或合作意向，可在项目中留言或通过邮箱联系：yuxuanliu0128@163.com。来信请提前注明来意。

🎉 Thrilled to share that our paper SMP-UWGS: Coupled Physics-Geometry Optimization for Scalable Multi-Partition Underwater 3D Reconstruction has been accepted to ECCV 2026!

🌊 The Challenge: Underwater 3D reconstruction struggles with light attenuation, backscattering, turbidity, and depth-dependent color distortion, making it hard to balance quality, color consistency, and training efficiency.

💡 Our Solution: We propose SMP-UWGS, an underwater 3DGS framework coupling physics modeling with geometric optimization. It uses multi-partition Gaussian representations for efficient large-scale optimization and a differentiable physics-based rendering module to jointly estimate optical parameters, accurately modeling depth-dependent attenuation and scattering.

📦 Code, paper, and project page coming soon. Stay tuned!

🙏 Acknowledgments:

• Guowei Ni for guidance on research direction

• Xiurui Guo for help with writing

• Chengwei Cao & Zhiwei Wu for valuable feedback during revision & rebuttal

• My partners at Willand for trust and guidance

• Shuai Liu & Ling Xue for funding support

• Junjun for her endless companionship and support throughout this journey

💬 I welcome any feedback! For research discussions or collaborations, please leave a comment or email me at mailto:yuxuanliu0128@163.com (please briefly state your purpose).
#ECCV2026 #Underwater3DReconstruction #3DGS #ComputerVision #DeepLearning
