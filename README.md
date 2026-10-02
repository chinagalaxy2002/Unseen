# FlashVTG DQ-CGP on Semantic Novelty × Event Existence Benchmark

本项目在 Charades-STA 衍生的语义新颖性 × 事件存在基准（`semantic_existence_v2`）上，对 **FlashVTG-GMR Baseline** 与引入 **DQ-CGP (v3)** 候选细化模块的模型进行了严格的 100 轮实验对比。

包含全部 5 个独立划分（`A1`, `A2_alt`, `A3`, `C1`, `C2_alt`）的训练脚本、推理评估脚本以及完整的客观评测数据。

---

## 1. 实验协议与基准说明

- **测试划分（5 个独立划分，独立训练）**：
  - **A1**: `put` / `take` 动作保留（Action holdout）
  - **A2_alt**: `drink` / `pour` 动作保留（Action holdout）
  - **A3**: `run` / `walk` 动作保留（Action holdout）
  - **C1**: `sit | bed/chair/couch` 构词保留（Composition holdout）
  - **C2_alt**: `open/close | box/cabinet` 构词保留（Composition holdout）
- **数据分区**：
  - $S^+$: 训练见过的语义，视频中存在目标事件
  - $S^-$: 训练见过的语义，视频中不存在目标事件
  - $U^+$: 训练未见的语义，视频中存在目标事件
  - $U^-$: 训练未见的语义，视频中不存在目标事件
- **协议约束**：
  - **训练集**：仅使用 $S^+ / S^-$
  - **模型选择（Checkpoint Selection）**：严格仅使用 Seen 验证集（$S^+ / S^-$）的平均时序定位指标 $(R1@0.7 + R1@0.5)/2$
  - **正式测试**：在 $S^+ / S^- / U^+ / U^-$ 上进行四象限与配对评测，$U^+ / U^-$ 绝不参与任何调参或选模。

---

## 2. 实验脚本与复现命令

### 2.1 环境准备
```bash
conda activate univtg
```

### 2.2 协议与代码自检
在正式训练前执行 10 步完整性自检（包括数据完整性、特征覆盖率、前向损失计算与反向传播）：
```bash
python scripts/sanity_check.py
```

### 2.3 单划分训练命令
以划分 `A1` 为例，从头训练 100 轮（默认随机种子 3407，早停关闭）：
```bash
bash scripts/train_dq_cgp_semantic_existence.sh A1
```
可通过环境变量指定其他划分和 GPU，例如在 GPU 1 上训练 `C1`：
```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/train_dq_cgp_semantic_existence.sh C1
```

### 2.4 推理与四象限评估命令
使用训练好的 seen 验证最佳 checkpoint 进行全量测试集推理，并运行四象限诊断与官方 GMR 评测：
```bash
bash scripts/infer_dq_cgp_semantic_existence.sh A1
```

### 2.5 双 GPU 自动并发与接力队列
支持在双卡上自动并发执行多划分训练与测试推理（任务自动认领与接力）：
```bash
bash scripts/start_multisplit_parallel.sh
```

---

## 3. 测试结果与客观数据对比

### 3.1 五划分完整诊断评测对比表

注：
- $\text{Gap} = \text{Seen AUROC} - \text{Unseen AUROC}$
- $\text{Gap Recovery} = \text{Baseline Gap} - \text{DQ-CGP Gap}$
- $\text{Matched PairAcc}$: 同视频、同来源的一对 $U^+ / U^-$ 样本的存在分数排序正确率。
- 所有数据来自各划分生成的 `diagnostics.json`。

| 划分 (Split) | 模型 (Model) | Seen AUROC | Unseen AUROC | $\Delta$ Unseen | Gap (Seen - Unseen) | Gap Recovery | Matched PairAcc | U+ raw R1@0.5 | U+ gated R1@0.5 | U+ FRR | U- RR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1** | Flash Baseline<br>**Flash DQ-CGP** | 0.8158<br>0.8151 | 0.5065<br>0.4946 | <br>-0.0119 | 0.3093<br>0.3205 | <br>-0.0112 | 55.61%<br>56.25% | 29.46%<br>25.38% | 23.66%<br>20.22% | 19.14%<br>17.42% | 20.82%<br>17.07% |
| **A2_alt** | Flash Baseline<br>**Flash DQ-CGP** | 0.7729<br>0.7659 | 0.5240<br>0.5016 | <br>-0.0224 | 0.2489<br>0.2643 | <br>-0.0154 | 50.63%<br>50.00% | 42.86%<br>45.24% | 42.86%<br>45.24% | 0.00%<br>0.00% | 0.96%<br>0.00% |
| **A3** | Flash Baseline<br>**Flash DQ-CGP** | 0.7422<br>0.7491 | 0.6139<br>0.6032 | <br>-0.0107 | 0.1283<br>0.1459 | <br>-0.0176 | 51.55%<br>49.22% | 46.88%<br>**56.25%** | 34.38%<br>**44.27%** | 23.44%<br>17.71% | 43.77%<br>28.62% |
| **C1** | Flash Baseline<br>**Flash DQ-CGP** | 0.7533<br>0.7443 | 0.5479<br>**0.6206** | <br>**+0.0728** | 0.2054<br>**0.1237** | <br>**+0.0817** | 55.56%<br>**65.28%** | 51.23%<br>51.23% | 48.77%<br>46.91% | 3.70%<br>8.02% | 3.70%<br>21.11% |
| **C2_alt** | Flash Baseline<br>**Flash DQ-CGP** | 0.6907<br>0.6908 | 0.5474<br>0.5107 | <br>-0.0366 | 0.1434<br>0.1800 | <br>-0.0367 | 54.55%<br>48.48% | 42.61%<br>42.61% | 15.65%<br>13.04% | 61.74%<br>59.13% | 64.57%<br>57.87% |

---

### 3.2 语义轴（Axis）均值汇总

| 语义轴 (Axis) | 划分集合 | 模型 (Model) | Mean Seen AUROC | Mean Unseen AUROC | $\Delta$ Unseen | Mean Gap | Mean Gap Recovery | Mean Matched PairAcc |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **动作轴 (Action)** | A1, A2_alt, A3 | Flash Baseline<br>**Flash DQ-CGP** | 0.7770<br>0.7767 | 0.5481<br>0.5331 | <br>-0.0150 | 0.2288<br>0.2436 | <br>-0.0147 | 0.5260<br>0.5182 |
| **构词轴 (Composition)** | C1, C2_alt | Flash Baseline<br>**Flash DQ-CGP** | 0.7220<br>0.7175 | 0.5476<br>**0.5656** | <br>**+0.0180** | 0.1744<br>**0.1518** | <br>**+0.0226** | 0.5505<br>**0.5688** |
| **整体均值 (Overall)** | 全部 5 划分 | Flash Baseline<br>**Flash DQ-CGP** | 0.7550<br>0.7530 | 0.5479<br>0.5461 | <br>-0.0018 | 0.2071<br>0.2069 | <br>+0.0002 | 0.5358<br>0.5385 |

---

### 3.3 官方 GMR 全测试集指标对比 (Official Test Metrics)

数据提取自各划分的 `official_test_metrics.json`：

| 划分 (Split) | 官方 AUROC (Base → DQ) | 官方 G-mIoU@1 (Base → DQ) | 官方 mAP (Base → DQ) |
| :--- | :---: | :---: | :---: |
| **A1** | 66.03% → **66.83%** (+0.80%) | 35.01% → **36.72%** (+1.71%) | 36.91% → **38.34%** (+1.43%) |
| **A2_alt** | 70.32% → 70.20% (-0.12%) | 38.98% → **39.37%** (+0.39%) | 37.06% → 36.22% (-0.84%) |
| **A3** | 71.36% → 70.66% (-0.70%) | 39.31% → 38.96% (-0.35%) | 37.62% → **37.71%** (+0.09%) |
| **C1** | 69.60% → 69.57% (-0.03%) | 39.02% → 37.49% (-1.53%) | 37.88% → **38.26%** (+0.38%) |
| **C2_alt** | 68.77% → **68.97%** (+0.20%) | 39.21% → 38.39% (-0.82%) | 38.35% → 38.33% (-0.02%) |

---

## 4. 原始测试数据文件索引

所有划分的原始评测输出 JSON 文件均保存在 `metrics/` 目录下并已纳入版本控制：

- 汇总对比表：[`metrics/comparison_table.json`](metrics/comparison_table.json)
- **A1**: [`metrics/A1/diagnostics.json`](metrics/A1/diagnostics.json), [`metrics/A1/official_test_metrics.json`](metrics/A1/official_test_metrics.json)
- **A2_alt**: [`metrics/A2_alt/diagnostics.json`](metrics/A2_alt/diagnostics.json), [`metrics/A2_alt/official_test_metrics.json`](metrics/A2_alt/official_test_metrics.json)
- **A3**: [`metrics/A3/diagnostics.json`](metrics/A3/diagnostics.json), [`metrics/A3/official_test_metrics.json`](metrics/A3/official_test_metrics.json)
- **C1**: [`metrics/C1/diagnostics.json`](metrics/C1/diagnostics.json), [`metrics/C1/official_test_metrics.json`](metrics/C1/official_test_metrics.json)
- **C2_alt**: [`metrics/C2_alt/diagnostics.json`](metrics/C2_alt/diagnostics.json), [`metrics/C2_alt/official_test_metrics.json`](metrics/C2_alt/official_test_metrics.json)
