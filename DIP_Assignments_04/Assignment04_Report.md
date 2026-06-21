# Assignment 4 — Simplified 3D Gaussian Splatting

> **姓名：**冯洋  
> **学号：**SC25005002
> **实验平台：** AutoDL 云服务器 / NVIDIA GeForce RTX 4090 D (24GB) / Ubuntu 22.04 / PyTorch 2.12 + CUDA 13

---

## 摘要

本实验实现了一个简化版 3D Gaussian Splatting（3DGS）管道，使用纯 PyTorch 从多视角图像重建 3D 场景。实验流程包括：(1) 使用 COLMAP 恢复相机参数和稀疏 3D 点；(2) 从稀疏点初始化 3D 高斯，通过可微光栅化和 α-blending 优化场景表示；(3) 与官方 3DGS 实现进行对比分析。实验在 Lego 数据集（100 张 800×800 多视角渲染图）上进行。简化版经过 200 轮训练，L1 Loss 从 0.1175 降至 0.0298；官方实现训练 7,000 轮，测试 PSNR 达到 25.77 dB，SSIM 达到 0.9461。对比分析揭示了 tile-based 光栅化、自适应密度控制和 CUDA 加速是官方实现性能显著优于简化版的核心因素。

---

## 1. 环境配置

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 D (24GB VRAM) |
| CPU | Intel Xeon (AutoDL 云实例) |
| 内存 | 32GB |
| 操作系统 | Ubuntu 22.04 |

### 1.2 软件环境

| 依赖 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch (Task 1-2) | 2.12.1 (CUDA 13) |
| PyTorch (Task 3) | 2.0.1 (CUDA 11.8) |
| OpenCV | 4.13.0 |
| COLMAP | 3.7 |
| NumPy | 1.26.4 |

### 1.3 环境配置注意事项

在配置过程中遇到以下问题并解决：

1. **libstdc++ 版本冲突**：系统 `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` 版本过旧（GLIBCXX_3.4.30），而 PyTorch 需要 GLIBCXX_3.4.31。通过创建符号链接 `ln -sf /root/miniconda3/lib/libstdc++.so.6.0.34 /root/miniconda3/lib/libstdc++.so.6` 并设置 `LD_LIBRARY_PATH` 解决。

2. **pytorch3d 依赖移除**：PyTorch 2.12 无对应 pytorch3d 预编译轮子，但 pytorch3d 仅在 `gaussian_model.py` 中用于 KNN 距离计算和 `data_utils.py` 中用于最远点采样（已注释）。使用 `torch.cdist` + `torch.topk` 替代 `pytorch3d.ops.knn.knn_points`，完全消除该依赖。

3. **COLMAP 安装**：系统 apt 安装的 COLMAP 与 conda 库存在冲突，最终使用 apt 版 COLMAP 搭配 CUDA 11.8 库路径，CPU 模式运行 SIFT。

---

## 2. Task 1 — Structure-from-Motion with COLMAP

### 2.1 方法

使用 COLMAP 对 Lego 数据集的 100 张多视角图像进行运动恢复结构（SfM），恢复相机内外参和稀疏 3D 点云，作为 3DGS 的初始化。

运行命令：

```bash
python mvs_with_colmap.py --data_dir data/lego
python debug_mvs_by_projecting_pts.py --data_dir data/lego
```

### 2.2 结果

| 指标 | 数值 |
|------|------|
| 输入图像 | 100 张 (800×800) |
| 恢复的 3D 点数 | **5,811** |
| Bundle Adjustment 残差 | 60,918 |
| BA 迭代次数 | 3 |
| 初始重投影误差 | 0.426 px |
| 最终重投影误差 | 0.426 px |
| 终止条件 | 收敛（Convergence） |
| 运行时间 | **0.46 分钟** |

### 2.3 重投影验证

将恢复的 5,811 个 3D 点重投影回 100 个视角进行验证，生成对比图保存于 `data/lego/projections/`。左半部分为原始图像，右半部分为 3D 点重投影（黑色背景上的彩色点）。点云密集覆盖了 Lego 模型的轮廓区域，但表面细节较为稀疏，这是仅使用 SfM 稀疏重建的预期结果。

### 2.4 产出文件

```
data/lego/sparse/0_text/
├── cameras.txt      # 相机内参（PINHOLE 模型）
├── images.txt       # 100 个视角的相机外参 (R, t)
└── points3D.txt     # 5,811 个稀疏 3D 点 + RGB
```

---

## 3. Task 2 — Simplified 3D Gaussian Splatting

### 3.1 3D 高斯参数初始化

根据论文公式 (6)，每个 3D 高斯的协方差矩阵由缩放矩阵 $S$ 和旋转矩阵 $R$ 构造：$\Sigma = R S S^T R^T$。需要以下可学习参数：

| 参数 | 维度 | 初始化方式 |
|------|------|-----------|
| Position $\mu$ | (N, 3) | SfM 3D 点坐标 |
| Rotation $R$ | (N, 4) | 单位四元数 $(1,0,0,0)$ |
| Scaling $S$ | (N, 3) | KNN 局部密度估计（K=50）|
| Opacity $o$ | (N, 1) | $\sigma(8.0) \approx 0.9997$ |
| Color $c$ | (N, 3) | SfM 点 RGB 颜色 |

初始缩放尺度范围：$[0.1054, 0.5884]$（根据 KNN 平均距离 × 2 计算）。

### 3.2 核心代码实现

按照论文公式实现了以下 5 个关键模块：

#### TODO 1：3D 协方差矩阵（`gaussian_model.py:114`）

$$\Sigma = R S S^T R^T$$

```python
Covs3d = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
```

#### TODO 2：透视投影雅可比矩阵（`gaussian_renderer.py:50-61`）

投影函数：$u = f_x \frac{x}{z} + c_x$，$v = f_y \frac{y}{z} + c_y$

$$J = \begin{bmatrix} \frac{f_x}{z} & 0 & -\frac{f_x \cdot x}{z^2} \\ 0 & \frac{f_y}{z} & -\frac{f_y \cdot y}{z^2} \end{bmatrix}$$

```python
J_proj[:, 0, 0] = fx / cam_z
J_proj[:, 0, 2] = -fx * cam_x / cam_z_sq
J_proj[:, 1, 1] = fy / cam_z
J_proj[:, 1, 2] = -fy * cam_y / cam_z_sq
```

#### TODO 3：相机空间协方差变换（`gaussian_renderer.py:63-64`）

$$\Sigma_{\text{cam}} = R \Sigma_{\text{world}} R^T$$

```python
covs_cam = R_exp @ covs3d @ R_exp.transpose(1, 2)
```

#### TODO 4：2D 高斯值计算（`gaussian_renderer.py:87-110`）

$$f(\mathbf{x}; \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) = \frac{1}{2\pi\sqrt{|\boldsymbol{\Sigma}_i|}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)\right)$$

使用 2×2 矩阵的显式逆矩阵公式（而非 `torch.inverse`）以提高效率：

$$\Sigma = \begin{bmatrix} a & b \\ b & d \end{bmatrix}, \quad \Sigma^{-1} = \frac{1}{ad - b^2}\begin{bmatrix} d & -b \\ -b & a \end{bmatrix}$$

#### TODO 5：α-Blending 渲染（`gaussian_renderer.py:149-152`）

$$\alpha_i = o_i \cdot f_i, \quad T_i = \prod_{j < i}(1 - \alpha_j), \quad C = \sum_i T_i \alpha_i \mathbf{c}_i$$

```python
T = torch.cumprod(1.0 - alphas + 1e-10, dim=0)
T = torch.cat([torch.ones(1, H, W), T[:-1]], dim=0)
weights = alphas * T
```

#### 数值稳定性优化

训练初期出现 NaN loss，定位到两个问题并解决：

1. **Jacobian z 值 clamp（`gaussian_renderer.py:54`）**：原代码使用 `clamp(min=1e-6)`，导致近相机点（深度 < 1）产生极大的 Jacobian 值（$f_x/z \approx 10^8$），进而协方差矩阵元素达 $10^{16}$ 级别，行列式计算时发生灾难性抵消。修改为 `clamp(min=1.0)` 以与 `depths.clamp(min=1.)` 保持一致，这些近点随后被 `valid_mask = (depths > 1.)` 正确滤除。

2. **Power 上界 clamp（`gaussian_renderer.py:106`）**：添加 `power = torch.clamp(power, max=0.0)` 防止浮点舍入误差导致 power > 0 进而 `exp()` 溢出。

### 3.3 训练配置

| 超参数 | 数值 |
|------|------|
| Epochs | 200 |
| Batch Size | 1 |
| 图像下采样因子 | 8×（800 → 100 像素） |
| 高斯点数量 | 5,811 |
| 优化器 | Adam |
| 学习率（Position） | $1.6 \times 10^{-5}$ |
| 学习率（Color） | 0.025 |
| 学习率（Opacity） | 0.05 |
| 学习率（Scaling） | 0.005 |
| 学习率（Rotation） | 0.001 |
| 梯度裁剪 | 1.0 |

### 3.4 训练结果

| 指标 | 数值 |
|------|------|
| 训练时间 | **~12 分钟**（200 epochs × 100 images） |
| 初始 L1 Loss | 0.1175 |
| 最终 L1 Loss | **0.0298** |
| Loss 下降幅度 | 74.6% |

**Loss 收敛曲线（每 epoch 末记录）：**

| Epoch | L1 Loss |
|-------|---------|
| 0 | 0.1175 |
| 1 | 0.0794 |
| ~65 | 0.0396 |
| ~130 | 0.0316 |
| 199 | **0.0298** |

训练过程中 Loss 稳定下降，前 65 轮收敛较快（从 0.12 降至 0.04），之后缓慢收敛至 0.03 附近。

### 3.5 可视化结果

每 epoch 保存 4 个固定视角的 GT vs Rendered 对比图（`data/lego/checkpoints/debug_images/epoch_*.png`），上方为 Ground Truth，下方为本模型渲染结果。训练完成后自动生成视角还原视频（`debug_rendering.mp4`），沿原始训练相机路径逐个视角渲染，左半部分为原始图像，右半部分为模型渲染图像。

---

## 4. Task 3 — 与官方 3DGS 实现对比

### 4.1 官方 3DGS 环境配置

由于官方 3DGS 代码中的 CUDA 扩展（`diff-gaussian-rasterization`、`simple-knn`）为 CUDA 11.x 时代编写，无法在新版 CUDA 13 下编译，因此**另开一台 AutoDL 实例**，镜像配置如下：

| 组件 | Task 1-2 环境 | Task 3 环境 |
|------|:--:|:--:|
| PyTorch | 2.12.1 | **2.0.1** |
| CUDA Toolkit | 13 | **11.8** |
| Python | 3.10 | 3.8 |

**编译要点**：
- 设置 `TORCH_CUDA_ARCH_LIST="8.9"` 针对 RTX 4090 D 的 Ada Lovelace 架构编译
- `diff-gaussian-rasterization` 子模块依赖 `glm` 库，需单独下载放入 `third_party/`
- `simple-knn` 子模块位于 GitLab，需从 `gitlab.inria.fr/bkerbl/simple-knn` 获取
- Python 代码与 CUDA 模块版本不匹配时需手动移除新增 API（如 `antialiasing`、`depth_image`）

### 4.2 RGBA 图像陷阱

训练初期 PSNR 仅 ~7 dB，远低于预期。经逐像素分析发现：

| 诊断 | 数值 |
|------|------|
| GT 像素 [200, 200] | [138, 100, 21] |
| Pred 像素 [200, 200] | [137, 99, 22] ✓ 近乎匹配 |
| GT 非零像素 | 38,793 |
| Pred 非零像素 | **112,430**（3 倍溢出） |

根因为 **Lego 数据集图像为 RGBA 格式**（4 通道），官方 3DGS 代码在加载时未正确处理 alpha 通道，导致像素值被错误解释。解决方法——将所有图像转为 RGB：

```python
img = Image.open(path)
if img.mode == 'RGBA':
    img.convert('RGB').save(path)
```

转换后 PSNR 从 6.70 dB **跃升至 26.05 dB**（提升 ~4 倍），验证了图像通道格式的关键影响。

### 4.3 渲染质量对比

使用 `--eval` 标志将数据集按 8:1 划分为训练集（87 张）和测试集（13 张）。

| 指标 | 简化版 (Task 2) | 官方 3DGS (Task 3) |
|------|:--:|:--:|
| **Train PSNR** | — | **25.90 dB** |
| **Test PSNR** | — | **25.77 dB** |
| **Train SSIM** | — | **0.9528** |
| **Test SSIM** | — | **0.9461** |
| 训练轮数 | 200 epochs | **7,000 iterations** |
| 初始高斯数 | 5,811 | 5,809 |
| 最终高斯数 | 5,811 | **250,205** |
| 图像分辨率 | 100×100 (8×下采样) | **400×400** |
| 训练 L1 Loss | **0.0298** (train) | 0.0117 (train) / 0.0122 (test) |

> **注**：简化版未计算 PSNR/SSIM（仅记录 L1 Loss），且运行在更低分辨率（100×100 vs 400×400），故直接数值对比有局限性。简化版若在 400×400 分辨率下训练，渲染每张图的时间将增长 16 倍。

### 4.4 训练速度对比

| 指标 | 简化版 (Task 2) | 官方 3DGS (Task 3) |
|------|:--:|:--:|
| 训练时间 | **~12 分钟** | **~2 分钟** |
| 迭代/轮数 | 200 epochs (20,000 张图) | 7,000 iterations |
| 吞吐量 | ~28 img/s | **~93 it/s** |
| 每张图处理时间 | ~36 ms | ~11 ms |
| 光栅化方式 | 朴素 $O(N \times H \times W)$ | **Tile-based CUDA** |

官方 3DGS 的 tile-based CUDA 光栅化器将图像划分为 16×16 的 tile，每个 tile 仅处理覆盖该区域的高斯，避免了对全图像素 × 全高斯的遍历。加之 CUDA 并行计算，训练速度差距达 **6 倍以上**。

### 4.5 显存占用对比

| 指标 | 简化版 (Task 2) | 官方 3DGS (Task 3) |
|------|:--:|:--:|
| 高斯数量 | **5,811** | 250,205 |
| 图像分辨率 | 100×100 | **400×400** |
| 显存占用（估计） | ~2-4 GB | ~4-8 GB |
| 内存效率 | 全高斯展开 $O(N \times H \times W)$ | **Tile-based 分块处理** |

简化版因分辨率低（100×100）且固定高斯数（5,811），显存占用相对较低。官方 3DGS 虽分辨率提升至 400×400 且高斯数膨胀至 25 万，但 tile-based 分块策略将每个 tile 的工作集压缩到常量级，避免了 $O(N)$ 的显存扩展。

### 4.6 自适应密度控制（Adaptive Density Control）

官方 3DGS 的核心创新之一是自适应密度控制：

| 阶段 | 高斯数量 |
|------|------|
| 初始化 | 5,809 |
| 7000 iterations 后 | **250,205**（43 倍增长）|

密度化策略：
- **Clone（克隆）**：对梯度大但尺寸小的高斯（欠重建区域），复制并沿梯度方向移动
- **Split（分裂）**：对梯度大且尺寸大的高斯（过重建区域），分裂为多个小高斯
- **Prune（剪枝）**：定期移除低透明度或过大的高斯，保持总数在可控范围

简化版缺少该机制，5,811 个固定高斯无法充分覆盖场景的细节区域，是渲染质量差距的主要来源。

### 4.7 差异来源分析

1. **光栅化效率（Tile-based vs Naive）**：官方实现的 tile-based 光栅化器（`diff-gaussian-rasterization`）将计算限制在每 tile 内，避免了 $O(N \times H \times W)$ 的全展开。简化版的朴素实现将 $N$ 个高斯展开为 $(N, H, W)$ 的张量，在 400×400 分辨率下会 $OOM$，因此被迫使用 100×100 下采样。

2. **自适应密度控制**：官方实现在训练过程中动态调整高斯数量（5,809 → 250,205），使得欠重建区域获得更多高斯覆盖，过重建区域被剪枝，大幅提升细节表达力。简化版固定高斯数，对纹理丰富区域覆盖不足。

3. **CUDA 加速**：官方实现将投影、排序、α-blending 全部用 CUDA kernel 实现，高度并行化。简化版使用 PyTorch 高级 API，引入大量中间张量分配和 Python 解释开销。

4. **球谐函数（SH）颜色建模**：官方实现使用 3 阶球谐函数（48 维特征）编码与视角相关的颜色，而简化版使用固定 RGB，无法建模高光等视角相关效果。

5. **图像格式敏感性**：本次实验发现官方 3DGS 对 RGBA 格式的输入图像处理异常（PSNR 从 26 dB 骤降至 7 dB），这是一个值得注意的工程细节，原因可能是 PIL 读取 RGBA 图像后的通道排列与代码预期不一致。

---

## 5. 总结

本实验从零实现了一个简化的 3D Gaussian Splatting 管道，完整覆盖了 SfM 相机恢复 → 3D 高斯初始化 → 可微光栅化 → α-blending 渲染的流程，并与官方 3DGS 实现进行了系统对比。在 Lego 数据集上的核心成果：

| 维度 | 简化版 (Task 2) | 官方 3DGS (Task 3) |
|------|:--:|:--:|
| 渲染质量 | L1=0.0298 (train) | PSNR=25.77 dB, SSIM=0.9461 |
| 训练时间 | ~12 min | **~2 min** |
| 高斯数量 | 5,811 (固定) | **250,205** (动态) |
| 图像分辨率 | 100×100 | **400×400** |

主要经验：

1. **数值稳定性至关重要**：3DGS 的投影协方差计算对近平面点非常敏感，Jacobian 的 z 值 clamp 需要与深度 mask 的阈值保持一致。FP32 精度下 $2\times2$ 矩阵的行列式计算 $ad-b^2$ 在大约 $10^{16}$ 数量级元素时发生灾难性抵消。

2. **Tile-based 光栅化是性能核心**：官方 3DGS 将图像分为 16×16 tile，每 tile 仅处理覆盖该区域的高斯，避免了简化版 $O(N \times H \times W)$ 的全展开。配合 CUDA 并行，达到了 6 倍以上的吞吐量差距。

3. **自适应密度控制是质量关键**：官方实现在训练中从 5,809 点扩展至 250,205 点（43 倍），通过 clone/split/prune 动态调整高斯分布。简化版固定高斯数无法充分覆盖场景细节。

4. **图像格式是隐藏陷阱**：RGBA 图像在官方代码中未正确处理 alpha 通道，导致 PSNR 从 26 dB 骤降至 7 dB。这是实际工程中容易忽略的细节。

5. **纯 PyTorch 实现的教学价值**：尽管性能远逊于官方实现，简化版以 ~400 行纯 Python 代码完整呈现了 3DGS 的数学原理，对于理解论文公式、调试数值问题、排除环境依赖有不可替代的价值。

## 6. 课程总结

作为一个传统工科的学生，这门课是最让我觉得我现在所学的内容是与现实接轨的，平常学习的力学课程上的内容与现实实践相去甚远。这门课的让我第一次学会了怎样去用Claude code搭配deepseek或是其他ai去完成一些网页上无法完成的事情，怎样用云服务器计算，怎样配置环境（这是对我来说最耗费时间的）。在这个过程中，我真正体会到了ai的强大，这门课程于我有着丰富的实践价值。
---

## 附录：文件结构

```
Assignment04/
├── data/lego/
│   ├── images/                     # 100 张输入图像（RGBA → RGB 转换后）
│   ├── images_rgba_backup/         # 原始 RGBA 图像备份
│   ├── input -> images/             # 官方 3DGS 期望的符号链接
│   ├── sparse/
│   │   ├── 0_text/                 # COLMAP SfM 输出（Task 1）
│   │   └── 0 -> 0_text/            # 官方 3DGS 格式符号链接
│   ├── projections/                # 重投影验证图（Task 1）
│   ├── checkpoints/
│   │   ├── checkpoint_000200.pt    # 最终训练模型（Task 2）
│   │   ├── debug_images/           # GT vs Rendered 对比图
│   │   └── debug_rendering.mp4     # 视角还原视频
│   └── output/
│       └── official_rgb/           # 官方 3DGS 输出（Task 3）
│           ├── point_cloud/
│           │   └── iteration_7000/
│           │       └── point_cloud.ply  # 250,205 个高斯
│           ├── train/ours_7000/     # 训练集渲染 87 views
│           └── test/ours_7000/      # 测试集渲染 13 views
├── gaussian_model.py               # 3D 高斯模型（已补全 TODO 1）
├── gaussian_renderer.py            # 可微渲染器（已补全 TODO 2-5）
├── data_utils.py                   # COLMAP 数据加载
├── train.py                        # 训练脚本
├── mvs_with_colmap.py              # COLMAP 管道脚本
├── debug_mvs_by_projecting_pts.py  # 重投影验证脚本
├── eval_official.py                # 官方 3DGS 指标计算（Task 3）
├── convert_rgba_to_rgb.py          # RGBA→RGB 转换工具
└── Assignment04_Report.md          # 本报告
```
