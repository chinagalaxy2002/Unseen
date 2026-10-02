你现在要继续一个已经完成 baseline 验证的研究项目。不要重新定义问题，不要重新设计 benchmark，也不要先提出新模型。

你要完成的任务只有一个：

# 核心实验目标

在我们构造的 **semantic novelty × event existence** 数据集上，已经观察到 **Moment-DETR-GMR 从 seen semantics 到 unseen semantics 的明显性能退化**。

现在只验证：

> **在完全相同的 semantic novelty × existence 协议下，给 Moment-DETR-GMR 加入 DQ-CGP 后，seen→unseen degradation 能否减缓或恢复。**

只做两个方法：

1. **LS-DQCGP**：/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-github-publish/ls_dq_cgp_lab
2. **DQ-CGPv3**：/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main/scripts/train_dq_cgp_v3.sh 

这两个方法是不一样的！可以认真阅读model文件。

DQ-CGP 源仓库：

`https://github.com/chinagalaxy2002/DQ-CGP`
这个仓库对应本地仓库为：/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main
！注意：所有需要做的实验都要在/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen2，请将两个仓库的代码先复制到这里，然后再修改。
DQ-CGPv3 原始训练入口：
`/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-main/scripts/train_dq_cgp_v3.sh`
LS-DQCGP 模型位置：`/home/guoxiangyu/VLMbasedIter_momentretrival/DQ-CGP-github-publish/ls_dq_cgp_lab`

主项目：
`https://github.com/chinagalaxy2002/GMR_Unseen`
这个仓库对应本地仓库为/home/guoxiangyu/paper/Openword/generalized-moment-retrieval

不要扩展为 event binding 机制研究，不要增加新的方法，不要自行加入 PCGrad、adapter、额外 calibration、额外 prompt 方法或新的 loss。

---

# 一、先恢复 GMR_Unseen 当前研究状态

开始写代码前，依次阅读：

1. `README.md`
2. `docs/PROJECT_HANDOFF.md`
3. `docs/reports/semantic_existence_multisplit_results.md`
4. `scripts/run_semantic_existence_100ep_tmux.sh`
5. `scripts/run_semantic_multisplit_100ep.sh`
6. `scripts/finalize_semantic_multisplit_group.sh`
7. `scripts/analyze_semantic_existence.py`
8. Moment-DETR-GMR 对应的 model / dataset / train / evaluate 实现

需要理解的核心协议是：

四象限：

- S+：seen semantic + event present
- S−：seen semantic + event absent
- U+：unseen semantic + event present
- U−：unseen semantic + event absent

训练：

- 只能使用 S+ / S−
- U+ / U− 不能参与训练

模型选择和 existence threshold：

- 只能使用 seen validation，即 S+ / S−
- U validation 不得参与 checkpoint selection
- U validation 不得参与 threshold calibration
- U validation/test 不得用于调超参数、选 epoch、选公式或选择模型变体

最终在 test 上统一评估 S+/S−/U+/U−。

研究核心不是“提高拒绝率”，而是：

- U+ 应被接受并准确定位
- U− 应被拒绝
- 这种能力应从 seen semantics 泛化到 unseen semantics

---

# 二、恢复 DQ-CGP 的两个方法

然后阅读 DQ-CGP：

## Experiment 1：LS-DQCGP

重点阅读：

- `README.md`
- `ls_dq_cgp_lab/cgp_module.py`
- `ls_dq_cgp_lab/ls_dq_cgp_model.py`
- `ls_dq_cgp_lab/train_ls_dq_cgp.py`
- `ls_dq_cgp_lab/run_experiment_with_exist.sh`
- `results/ls_dq_cgp_exist_seed2023/RESULTS.md`

我们要移植的是：

**LS-DQCGP + existence head**

不是 localization-only 版本。

保留方法本身的核心定义，包括：

- native D1 temporal binding
- candidate-specific local visual context
- RCG
- BPS
- FRF
- late semantic matcher
- independent GMR existence head

初始方法参数直接采用 DQ-CGP 仓库已经使用的固定配置，不允许根据 U 结果重新搜索：

- `native_bind_coef = 0.2`
- `num_basis = 16`
- `prompt_length = 6`
- `router_hidden_dim = 256`
- `frf_hidden_dim = 512`
- `temperature = 1.0`

## Experiment 2：DQ-CGPv3

重点阅读：

- `scripts/train_dq_cgp_v3.sh`
- `configs/moment_detr_gmr/model/moment_detr_vmr_cgp_v3.yml`
- `models/moment_detr_gmr/moment_detr.py`
- DETR Query CGP 相关模块和 loss 实现

保留 DQ-CGPv3 原方法定义：

- candidate-specific DETR query adaptation
- binding loss
- route loss
- D1→D2 residual injection

固定原始方法参数：

- `query_cgp_num_basis = 16`
- `query_cgp_prompt_length = 6`
- `query_cgp_router_hidden_dim = 256`
- `query_cgp_frf_hidden_dim = 512`
- `query_cgp_temperature = 1.0`
- `query_cgp_beta = 0.05`
- `query_cgp_binding_loss_coef = 0.2`
- `query_cgp_route_loss_coef = 0.01`

不要重新设计 DQ-CGPv3。

---

# 三、这里验证的是 transfer，不是重新优化 DQ-CGP

非常重要：

DQ-CGP 原仓库是在 Soccer-GMR 上开发的。

现在的问题不是：

“怎样把 DQ-CGP 在 Charades 上调到最好？”

而是：

> **把已经定义好的 DQ-CGP 方法迁移到 GMR_Unseen 的严格 semantic-novelty protocol 后，它是否能够减缓 Moment-DETR-GMR 已经观察到的 seen→unseen degradation？**

因此：

- 方法结构尽量保持 DQ-CGP 原定义
- 训练和评价协议必须服从 GMR_Unseen
- 不允许根据 U 表现修改 DQ-CGP 超参数
- 不允许根据 test 结果决定继续改什么

---

# 四、训练协议必须与原 Moment-DETR-GMR baseline 对齐

不要直接复制 Soccer-GMR 的 400 epoch 训练协议。

GMR_Unseen 第二阶段正式 Moment baseline 使用：

- seed = `3407`
- 100 epochs
- no early stopping
- `max_es_cnt = -1`
- batch size = 16
- eval batch size = 16
- learning rate 使用原 Moment baseline 设置
- 相同 CLIP text features
- 相同 CLIP video features
- 相同 SlowFast features
- 相同 train / val / test split
- 相同 existence head
- 相同 existence loss
- 相同 optimizer/scheduler，除非 DQ 方法本身严格要求改变
- checkpoint selection 只能基于原有 seen-validation 规则

不要因为 DQ-CGP 原仓库使用 400 epoch + early stopping，就给它额外训练预算。

本实验首先要求 **matched protocol**。

---

# 五、特别控制 LS-DQCGP 的 saliency 差异

DQ-CGP 原始 LS-DQCGP + Exist Soccer 实验设置中存在：

`mr_only = False`
`lw_saliency = 1`

而当前 GMR_Unseen 的 Moment-DETR-GMR 正式 baseline 是：

`mr_only = True`
`lw_saliency = 0`

本实验不能混入额外 saliency supervision。

因此 LS-DQCGP 在 GMR_Unseen 上必须改成：

`use_exist_head = True`
`keep_empty_gt = True`
`mr_only = True`
`lw_saliency = 0`

即：

- 保留 existence training
- 保留 S− empty-GT samples
- 不增加 baseline 原本没有的 saliency supervision

否则实验无法回答 DQ-CGP 本身是否减缓退化。

DQ-CGPv3 同样必须保持这一 matched training protocol。

---

# 六、Text feature 不要重新提取

GMR_Unseen 当前 Charades text NPZ 保存的是已经裁剪后的：

`last_hidden_state`

而不是 DQ-CGP 某些脚本期待的：

`attention_mask`

不要为了 DQ-CGP 重新提取文本特征，因为这会改变 baseline 输入。

对于 LS-DQCGP 和 DQ-CGPv3：

优先使用原 dataloader 已有的 `src_txt_mask` 作为有效 token mask。

如果代码支持：

`src_txt_semantic_mask = None`

则让方法退回使用：

`src_txt_mask`

不要因为 semantic mask 接口不同而改变 CLIP text features。

---

# 七、正式实验范围

主实验只针对：

**Moment-DETR-GMR**

不用做 QD-DETR。
不用做 FlashVTG。

正式 semantic splits 使用已经冻结的五组：

- A1：put / take
- A2_alt：drink / pour
- A3：run / walk
- C1：sit + bed/chair/couch compositions
- C2_alt：open/close + box/cabinet compositions

每个 split 独立训练。

最终方法矩阵：

| Split | Existing baseline | LS-DQCGP | DQ-CGPv3 |
|---|---|---|---|
| A1 | Moment-DETR-GMR | train | train |
| A2_alt | Moment-DETR-GMR | train | train |
| A3 | Moment-DETR-GMR | train | train |
| C1 | Moment-DETR-GMR | train | train |
| C2_alt | Moment-DETR-GMR | train | train |

现有 Moment baseline 已经完成。

除非发现 baseline artifact 无法复用，否则不要为了这个实验重新训练额外 baseline。

因此主要新增训练共：

**5 splits × 2 methods = 10 runs**

全部使用 seed 3407。

在正式十个训练前，先只在 A1 上完成两个方法的 smoke test。

Smoke test 只检查工程正确性，不能根据 U 指标修改方法。

---

# 八、代码隔离

所有新代码、wrapper、配置、运行脚本、日志和报告放入新的独立目录，例如：

`experiments/dq_cgp_semantic_generalization/`

建议结构：

`experiments/dq_cgp_semantic_generalization/`
- `README.md`
- `SOURCE_AUDIT.md`
- `COMPATIBILITY.md`
- `EXPERIMENT_PLAN.md`
- `ls_dqcgp/`
- `dq_cgp_v3/`
- `scripts/`
- `configs/`
- `runs/`
- `reports/`

不要修改：

- 原 semantic release
- 已完成 baseline 结果
- correspondence_generalization
- event_binding_generalization
- 原正式结果文件

如果必须复用 DQ-CGP 源码：

优先复制最少必要模块到新目录，或者做 wrapper。

记录：

- DQ-CGP source repository
- source commit SHA
- 每个复制文件的来源

---

# 九、训练前必须完成的工程检查

两个方法都必须检查：

1. S+ 可以正常 forward/backward
2. S− empty GT 可以正常 forward/backward
3. S− 不产生 span/binding 的非法 loss
4. existence loss 在 S− 上正常存在
5. mixed S+/S− batch 正常训练
6. Hungarian matcher 能处理 empty GT
7. 没有 NaN / Inf
8. prediction 中包含 `pred_exist_score`
9. prediction 中保留 raw localization windows
10. existence gate 前后的 localization 可以分别评估

对于 LS-DQCGP：

确认 native binding loss 在没有 GT span 的 S− 上为 0，而 existence loss 仍然工作。

对于 DQ-CGPv3：

确认 query-CGP binding loss 同样正确处理 empty-GT S−。

如果这里有 bug，只修工程兼容问题，不改变方法定义。

---

# 十、模型选择绝对不能看 U

训练过程中 validation 必须只读取：

`val_seen.jsonl`

或者等价的 S+/S− validation view。

不允许把完整 val 中的 U+/U− 输入 checkpoint selection。

如果训练代码内部默认根据 MR mAP 保存 best checkpoint，需要核对它使用的数据是否严格是 seen validation。

采用与原 Moment baseline 一致的 checkpoint selection 规则。

不要为了 DQ-CGP 使用新的 U-aware selection metric。

---

# 十一、最终评价必须使用 GMR_Unseen 原评价协议

最终 evaluator 优先复用：

`scripts/analyze_semantic_existence.py`

以及项目已有 multisplit finalize/evaluation pipeline。

不要以 DQ-CGP 的 Soccer-GMR evaluator 作为主评价结果。

每个方法、每个 split 最少输出：

## Existence

- seen AUROC
- unseen AUROC
- seen→unseen AUROC gap

定义：

`exist_gap = seen_AUROC - unseen_AUROC`

## Positive localization

分别报告：

- S+ raw R@1 IoU 0.5
- U+ raw R@1 IoU 0.5
- S+ gated R@1 IoU 0.5
- U+ gated R@1 IoU 0.5

## Gate behavior

- S+ false-refusal rate
- U+ false-refusal rate
- S− rejection rate
- U− rejection rate

## Matched unseen pairs

- matched U+/U− pair accuracy

所有 existence threshold 必须来自：

**seen validation S+/S−**

不能从 U 或 test 选择。

---

# 十二、怎样判断“减缓退化”

本实验最重要的比较不是单独看：

`gap 是否变小`

而是看 **unseen performance 是否真正恢复**。

对每个 split、每个方法计算：

`gap_baseline = seen_AUROC_baseline - unseen_AUROC_baseline`

`gap_method = seen_AUROC_method - unseen_AUROC_method`

可以辅助计算：

`gap_reduction = gap_baseline - gap_method`

但是不能仅凭 gap reduction 宣称成功。

例如：

Baseline：

seen = 0.87
unseen = 0.58
gap = 0.29

Method：

seen = 0.65
unseen = 0.57
gap = 0.08

这不是恢复。

因为 gap 是通过破坏 seen performance 缩小的。

我们真正希望看到：

- unseen AUROC 上升
- gap 缩小
- seen AUROC 基本保持

---

# 十三、主要判断标准

对于 LS-DQCGP 和 DQ-CGPv3，分别判断：

### 1. existence generalization

最重要：

`unseen_AUROC_method - unseen_AUROC_baseline`

是否为正。

同时：

`exist_gap_method < exist_gap_baseline`

并确认这种 gap reduction 不是由 seen AUROC 大幅下降造成。

### 2. U+ localization

检查：

`U+ raw R1@0.5`

是否保持或改善。

这用于防止方法只是改变 rejection，而破坏本来可以定位的 U+。

### 3. end-to-end

检查：

`U+ gated R1@0.5`

是否保持或改善。

### 4. U− rejection

检查：

U− rejection rate 是否没有明显恶化。

禁止通过“几乎所有 unseen query 都接受”来换取 U+ 改善。

### 5. seen guardrail

S+/S− 上的 existence 和 localization 不应出现明显性能崩溃。

---

# 十四、不要把什么写成成功

以下情况不算减缓退化：

1. seen AUROC 大幅下降，因此 gap 人为缩小
2. 只提高 U+ 接受率，但 U− rejection 崩溃
3. 只提高 U− rejection，但 U+ false refusal 上升
4. unseen AUROC 提高，但 U+ raw localization 明显下降
5. raw localization 提高，但 gate 把正确 U+ 全部拒绝
6. 只在一个 split 上偶然改善，就写成通用恢复
7. 根据 U/test 调过参数之后再报告提升

---

# 十五、五个 split 的最终汇总

除了逐 split 结果，还要像原 multisplit report 一样给出：

## Action splits 等权平均

A1 / A2_alt / A3

分别计算：

- baseline seen AUROC
- baseline unseen AUROC
- LS seen / unseen AUROC
- DQ-CGPv3 seen / unseen AUROC
- baseline gap
- LS gap
- V3 gap
- unseen AUROC delta
- gap reduction

## Composition splits 等权平均

C1 / C2_alt

同样计算。

同时报告：

- 5 个 split 中，有多少个 split 的 unseen AUROC 高于 baseline
- 有多少个 split 同时满足：
  - unseen AUROC ↑
  - gap ↓
  - seen 没有明显下降

不要只报告平均值而隐藏单 split 失败。

---

# 十六、实验的科学问题只写这一句

最终报告围绕：

> **Can DQ-CGP reduce the seen-to-unseen degradation of Moment-DETR-GMR under the semantic novelty × event existence protocol?**

不要把实验扩大解释成：

- event binding 已被证明是根因
- DQ-CGP 解决了所有 unseen GMR
- 所有 backbone 都适用
- 所有开放词汇视频定位都适用

本实验只能回答：

**在当前冻结的 Charades-STA semantic novelty × existence benchmark 上，LS-DQCGP / DQ-CGPv3 是否能够减缓 Moment-DETR-GMR 已观察到的 seen→unseen generalization degradation。**

---

# 十七、你的实际工作顺序

严格按以下顺序执行：

1. 阅读并恢复 GMR_Unseen baseline protocol
2. 阅读 LS-DQCGP 和 DQ-CGPv3 源实现
3. 写 `SOURCE_AUDIT.md`
4. 写 `COMPATIBILITY.md`
5. 明确哪些地方必须为 Charades/GMR_Unseen 做接口适配
6. 确认除了必要兼容之外没有改变方法
7. 冻结两个方法配置
8. 完成 empty-GT / existence / binding 单元测试
9. A1 上分别做 LS-DQCGP 和 DQ-CGPv3 smoke test
10. smoke 通过后执行 A1 正式训练
11. 不根据 A1 U/test 修改超参数
12. 使用同一冻结配置完成 A2_alt、A3、C1、C2_alt
13. 用 GMR_Unseen 原 evaluator 完成四象限评价
14. 和现有 Moment baseline 一一对比
15. 输出逐 split 和 action/composition 等权汇总
16. 最终只回答两个方法是否减缓 degradation

如果运行中发现方法与数据接口存在工程不兼容，优先做最小兼容修复并记录，不要趁机重新设计模型。

如果某个方法失败，保留负结果，不通过查看 U/test 后重新扫描超参数挽救。