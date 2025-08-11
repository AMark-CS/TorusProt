# Torsion Angle Flow Matching for Protein Structure Generation

这个项目在 FoldFlow 的基础上实现了基于二面角和键角的蛋白质结构生成模型。主要创新点包括：

1. **混合几何表示**：二面角使用环面(Torus)流匹配，键角使用欧几里得流匹配
2. **NERF重建**：从角度空间重建笛卡尔坐标
3. **物理约束保持**：保持蛋白质的几何约束和周期性

## 🏗️ 架构概述

### 数据表示
- **二面角 (φ, ψ, ω)**: 转换为环面坐标 (cos θ, sin θ)
- **键角**: N-CA-C, CA-C-N, C-N-CA 角度
- **键长**: N-CA, CA-C, C-N 距离（可选，通常使用标准值）

### 模型组件
1. **TorusFlowMatcher**: 处理二面角的环面流匹配
2. **EuclideanFlowMatcher**: 处理键角和键长的欧几里得流匹配
3. **MixedFlowMatcher**: 组合两种流匹配器
4. **DifferentiableNERF**: 从角度重建坐标

## 📦 新增文件

```
foldflow-mace/
├── foldflow/data/
│   └── torsion_angle_loader.py          # 二面角数据加载器
├── foldflow/models/
│   ├── torus_flow.py                    # 环面流匹配模型
│   └── nerf_reconstruction.py          # NERF坐标重建
├── runner/
│   ├── train_torsion.py                 # 二面角训练脚本
│   └── config/torsion_flow.yaml        # 训练配置
└── test_torsion_flow.py                 # 测试脚本
```

## 🚀 快速开始

### 1. 环境准备
确保已安装 FoldFlow 的依赖环境，包括：
- PyTorch >= 1.12
- biotite (用于蛋白质结构处理)
- wandb (可选，用于实验跟踪)

### 2. 测试安装
```bash
cd /storage2/hechuan/code/foldflow-mace
python test_torsion_flow.py
```

这会运行各个组件的单元测试，确保功能正常。

### 3. 数据准备
确保数据路径正确设置：
```yaml
# runner/config/torsion_flow.yaml
data:
  csv_path: /storage2/hechuan/code/FoldFlow-0.2.0/data/metadata_one.csv
  cluster_path: /storage2/hechuan/code/FoldFlow-0.2.0/data/clusters-by-entity-30.txt
```

### 4. 开始训练
```bash
# 在第二块GPU上训练，避免显存问题
CUDA_VISIBLE_DEVICES=1 python runner/train_torsion.py

# 或者使用特定配置
CUDA_VISIBLE_DEVICES=1 python runner/train_torsion.py \
  --config-name torsion_flow \
  experiment.batch_size=2 \
  data.filtering.max_len=150
```

## ⚙️ 配置说明

### 关键参数
```yaml
# 模型配置
model:
  torus_hidden_dim: 256        # 环面流隐藏层维度
  torus_layers: 6              # 环面流层数
  euclidean_hidden_dim: 256    # 欧几里得流隐藏层维度
  euclidean_layers: 4          # 欧几里得流层数

# 流匹配配置
flow_matcher:
  torus_sigma: 0.1             # 环面噪声标准差
  euclidean_sigma: 0.2         # 欧几里得噪声标准差
  num_sampling_steps: 100      # 生成时的积分步数

# 损失权重
loss:
  torus_weight: 1.0            # 二面角损失权重
  bond_angle_weight: 1.0       # 键角损失权重
  bond_length_weight: 0.1      # 键长损失权重（通常较小）
```

### 显存优化
```yaml
experiment:
  batch_size: 2                # 小批次大小
  num_loader_workers: 2        # 数据加载进程数

data:
  filtering:
    max_len: 150               # 序列长度上限（关键！）
    min_len: 50                # 序列长度下限

hardware:
  mixed_precision: true        # 混合精度训练
```

## 🔧 显存管理

### 问题诊断
如果遇到 CUDA OOM 错误：

1. **降低序列长度**：这是最有效的方法
   ```yaml
   data.filtering.max_len: 100  # 从150降到100
   ```

2. **减小批次大小**：
   ```yaml
   experiment.batch_size: 1     # 最小值
   ```

3. **减少模型复杂度**：
   ```yaml
   model:
     torus_hidden_dim: 128      # 从256降到128
     torus_layers: 4            # 从6降到4
   ```

### 显存监控
```bash
# 实时监控GPU使用情况
watch -n1 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 1"
```

## 📊 实验跟踪

### Wandb集成
```python
# 自动初始化（如果配置了use_wandb: true）
wandb.init(
    project="foldflow-torsion",
    name="torsion_flow_experiment",
    config=config_dict
)
```

### 关键指标
- `train/total_loss`: 总训练损失
- `train/torus_loss`: 二面角损失
- `train/bond_angle_loss`: 键角损失
- `val/total_loss`: 验证损失

## 🧪 结果验证

### 生成结构
训练完成后，模型会自动生成样本结构：
```python
# 在训练脚本中自动调用
samples = experiment.sample_structures(num_samples=4)
coords = samples['coordinates']  # [4, N*3, 3] 笛卡尔坐标
phi = samples['phi']             # [4, N] phi角度
psi = samples['psi']             # [4, N] psi角度
```

### 评估指标
- **RMSD**: 与参考结构的均方根偏差
- **GDT**: 全局距离测试得分
- **Ramachandran**: 二面角分布合理性

## 🐛 常见问题

### 1. 数据加载失败
```
FileNotFoundError: CSV file not found
```
**解决方案**: 检查 `csv_path` 路径是否正确

### 2. 显存不足
```
CUDA out of memory
```
**解决方案**: 降低 `max_len` 和 `batch_size`

### 3. 角度转换错误
```
Unexpected shape for dihedral angles
```
**解决方案**: 确保输入坐标为 N-CA-C 模式，长度为3的倍数

### 4. 模型收敛慢
**可能原因**:
- 学习率过高/过低
- 损失权重不平衡
- 噪声水平不合适

**解决方案**:
```yaml
experiment:
  learning_rate: 5e-5          # 降低学习率
loss:
  torus_weight: 2.0            # 调整权重比例
flow_matcher:
  torus_sigma: 0.05            # 降低噪声
```

## 🎯 进一步改进

### 1. 性能优化
- [ ] 实现 Flash Attention
- [ ] 使用 gradient checkpointing
- [ ] 优化 NERF 重建速度

### 2. 模型改进
- [ ] 添加侧链预测
- [ ] 集成蛋白质序列信息
- [ ] 多尺度流匹配

### 3. 评估增强
- [ ] 添加更多物理约束
- [ ] 实现自动评估流水线
- [ ] 与现有方法对比

## 📚 相关论文

1. **Flow Matching**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
2. **FoldFlow**: [FoldFlow: Flow Matching for Protein Structure Generation](https://arxiv.org/abs/2302.12931)  
3. **NERF**: [NERF: Neural Extension Reference Frame](https://academic.oup.com/bioinformatics/article/13/3/291/423201)
4. **环面几何**: [Manifold Learning on the Torus](https://ieeexplore.ieee.org/document/8417842)

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目基于 FoldFlow 的许可证条款。详见原项目的 LICENSE 文件。
