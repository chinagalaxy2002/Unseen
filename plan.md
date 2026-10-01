你现在要接手一个已有研究项目，并完成一个新的、范围严格受限的实验。
禁止修改本地已有仓库任何代码！
请不要从零重新定义问题，不要先提出新模型，也不要把实验扩展成机制研究。
在修改代码、启动训练之前，必须先恢复两个仓库当前的状态并理解本次实验唯一的研究目标。

============================================================
一、需要阅读的两个仓库
============================================================

【研究 benchmark / 数据 / baseline】
https://github.com/chinagalaxy2002/GMR_Unseen
这个仓库对应本地仓库为/home/guoxiangyu/paper/Openword/generalized-moment-retrieval

【待测试方法：FlashVTG-GMR + DQ-CGP】
https://github.com/chinagalaxy2002/Falshvtg-gmr-DQ-CGP.git
这个仓库对应本地仓库为：/home/guoxiangyu/VLMbasedIter_momentretrival/Falshvtg-gmr-DQ-CGP
！注意：所有需要做的实验都要在/home/guoxiangyu/VLMbasedIter_momentretrival/Unseen，请将两个仓库的代码先复制到这里，然后再修改。

数据在：/home/guoxiangyu/paper/Openword/data
============================================================
二、本次实验唯一最重要的目标
============================================================

我的目标非常简单：

我已经基于 Charades-STA 构造了一个新的 semantic novelty × event existence 数据集，
并已经观察到 FlashVTG-GMR 在 seen semantics 和 unseen semantics 之间存在明显泛化退化。

现在我想测试：

    把现有的 DQ-CGP 方法应用到这个新的数据集上以后，
    能不能减缓 FlashVTG-GMR 从 seen semantics 到 unseen semantics 的性能退化？

本次实验最核心的问题只有这一条。

不要把目标扩大成：

- 证明 event binding 是根本机制；
- 设计一个新的 SOTA 模型；
- 解释所有退化原因；
- 提高某一个固定阈值下的拒绝率；
- 解决整个 generalized moment retrieval；
- 重新设计数据集；
- 修改 benchmark 定义。

DQ-CGP 在这里就是一个“已有方法迁移实验”。

我们只想知道：

    原来的 FlashVTG-GMR 有明显 seen → unseen degradation，
    加 DQ-CGP 之后，这个 degradation 是否变小。

============================================================
三、原项目的问题定义
============================================================

GMR_Unseen 研究 semantic novelty × event existence。

四种测试情况：

S+：训练见过的语义 + 事件在视频中存在
S-：训练见过的语义 + 事件在视频中不存在

U+：训练没见过的语义 + 事件在视频中存在
U-：训练没见过的语义 + 事件在视频中不存在

训练：

    只能使用 S+ / S-

验证和模型选择：

    只能使用 seen validation
    即 S+ / S-

最终测试：

    S+ / S- / U+ / U-

U+ 和 U- 不能用于：

- checkpoint selection
- hyperparameter tuning
- threshold tuning
- calibration fitting
- model variant selection

核心原则：

    unseen != absent

这里的 unseen 是 downstream-task-training-unseen，
不是声称 CLIP / SlowFast 等预训练模型从未接触这些语义。


============================================================
四、已经确认存在的 baseline degradation
============================================================

GMR_Unseen 第二阶段已经冻结了五个语义划分：

A1：put / take
A2_alt：drink / pour
A3：run / walk
C1：sit | bed/chair/couch
C2_alt：open/close | box/cabinet

每个 split 独立构造训练集、独立训练。

FlashVTG-GMR baseline 的结果如下：

Split      Seen AUROC   Unseen AUROC   Seen-Unseen Gap
A1         0.8158       0.5065         0.3093
A2_alt     0.7729       0.5240         0.2489
A3         0.7422       0.6139         0.1283
C1         0.7533       0.5479         0.2054
C2_alt     0.6907       0.5474         0.1433

五个 split 上：

    Seen AUROC > Unseen AUROC

这是本次实验要尝试缓解的 degradation。

请始终把“baseline degradation”定义成：

    gap = seen_AUROC - unseen_AUROC


============================================================
五、本实验怎样定义“缓解退化”
============================================================

对于每个 split，计算：

    baseline_gap
      = baseline_seen_AUROC
      - baseline_unseen_AUROC

    dq_cgp_gap
      = dq_seen_AUROC
      - dq_unseen_AUROC

    gap_recovery
      = baseline_gap
      - dq_cgp_gap

如果：

    gap_recovery > 0

说明 seen-unseen gap 变小。

但是：

“gap 变小”本身不够。

以下情况不能被认为是真正缓解：

Baseline:
    seen = 0.80
    unseen = 0.50
    gap = 0.30

DQ-CGP:
    seen = 0.55
    unseen = 0.50
    gap = 0.05

这种情况只是 seen 性能崩了，不算泛化恢复。

因此本次实验最重要的判断原则是：

    1. unseen AUROC 应该上升；
    2. seen-unseen AUROC gap 应该下降；
    3. seen AUROC 不能因为明显下降而人为造成 gap 缩小。

请优先报告这三个量：

    Δ unseen AUROC
    Δ seen AUROC
    Δ gap

其中：

    Δ unseen AUROC
      = DQ unseen AUROC
      - baseline unseen AUROC

    Δ seen AUROC
      = DQ seen AUROC
      - baseline seen AUROC

    Δ gap
      = DQ gap
      - baseline gap

注意：

    Δ gap < 0

才表示 gap 被减缓。

也可以同时报告：

    gap_recovery = - Δ gap

此时：

    gap_recovery > 0

表示恢复。


============================================================
六、其他指标的地位
============================================================

本次项目的主要成功标准不是 localization，
也不是固定阈值 rejection rate。

U+ raw R@1@0.5、
U+ gated R@1@0.5、
U+ FRR、
U- RR、
PairAcc、
官方 GMR metrics

都可以继续计算和保存。

但是它们在本次实验中主要是：

    sanity check / diagnostic metrics

不要因为某个 rejection rate 提升就宣称实验成功。

本实验的核心结论只围绕：

    FlashVTG-GMR seen→unseen existence AUROC degradation
    是否因为 DQ-CGP 而得到缓解。


============================================================
七、为什么测试 DQ-CGP
============================================================

本次不需要先证明 DQ-CGP 对应某个机制假设。

DQ-CGP 只是一个已有方法候选。

待测试仓库：

https://github.com/chinagalaxy2002/Falshvtg-gmr-DQ-CGP.git

当前重点使用：

    models/flashvtg_dq_cgp_v3_gmr/

它是在 FlashVTG-GMR 上加入：

- candidate-wise temporal binding
- query-conditioned global/local prototypes
- sparse basis routing
- prompt bank
- FRF residual

的已有实现。

这个方法已经在原来的 Soccer-GMR 设置中使用过。

本次实验不是重新开发 DQ-CGP。

第一阶段应该：

    尽可能保持 DQ-CGP v3 原始结构和超参数，
    只把它移植到 GMR_Unseen 的 Charades semantic-existence protocol。

这样我们测试的是：

    method transfer

而不是：

    在新 benchmark 上重新为 DQ-CGP 设计一个新版本。


============================================================
八、第一阶段禁止修改的内容
============================================================

第一轮实验不要修改 DQ-CGP architecture。

优先保留当前 v3 参数：

    dq_cgp_num_basis = 16
    dq_cgp_prompt_length = 6
    dq_cgp_routing_topk = 4
    dq_cgp_point_mixture_ratio = 0.10
    dq_cgp_beta = 0.05

    dq_cgp_binding_loss_coef = 0.05
    dq_cgp_route_loss_coef = 0.01
    dq_cgp_relation_loss_coef = 0.02

特别是：

    dq_cgp_refine_exist = False

第一轮保持 False。

不要为了让结果更好，在 unseen test 上调：

- num_basis
- top-k
- beta
- loss coefficient
- existence head
- threshold
- calibration
- training epoch
- routing design

第一阶段的问题只是：

    当前已有 DQ-CGP v3 原样迁移过来，
    是否已经能够减缓这个 benchmark 上的 degradation？


============================================================
九、需要匹配的 FlashVTG baseline protocol
============================================================

不要直接使用 Soccer-GMR 的训练脚本参数。

本实验必须匹配 GMR_Unseen 中已经使用过的
FlashVTG-GMR semantic-existence protocol。

关键参数：

    dset_name = charadesSTA
    ctx_mode = video_tef

    v_feat_dim = 2816
    t_feat_dim = 512

    max_q_l = 40
    max_v_l = 200
    clip_length = 1
    max_windows = 5

    lr = 3e-5
    weight_decay = 1e-4

    epochs = 100
    early stopping = disabled

    batch size = 8
    seed = 3407

    hidden_dim = 256
    dim_feedforward = 1024
    enc_layers = 3
    t2v_layers = 6
    dummy_layers = 2
    nheads = 8

    use_exist_head = True
    exist_pool = mean
    exist_loss_coef = 1.0

    mr_only = True
    eval_full_only = True

    nms_thd = -1

请直接去 GMR_Unseen：

    scripts/run_semantic_existence_100ep_tmux.sh

核对 FlashVTG 的真实训练命令。

不要凭这个 prompt 猜参数。

以仓库实际脚本为最终依据。


============================================================
十、数据和特征
============================================================

使用 GMR_Unseen 已经冻结的：

    data/release/semantic_existence_v2/

划分：

    A1
    A2_alt
    A3
    C1
    C2_alt

每个 split：

    train.jsonl
        只含该 split 允许的 S+ / S-

    val_seen.jsonl
        只允许用于 checkpoint / threshold 等 seen-side selection

    test.jsonl
        含 S+ / S- / U+ / U-

使用与原 FlashVTG-GMR baseline 相同的：

    CLIP video feature
    SlowFast video feature
    CLIP text feature

不要重新生成一套不同编码器的 feature，
除非确认现有 DQ-CGP 实现存在技术上无法使用的原因。

如果 feature 结构兼容，直接复用。


============================================================
十一、非常重要：negative sample 的处理
============================================================

需要确认：

S-：

    relevant_windows = []

必须：

    参与 existence BCE

但不能得到伪 localization GT。

请核对 DQ-CGP 当前代码。

目前预期行为是：

    exist_label > 0.5
        才计算 MR / DQ-CGP localization losses

negative：

    只保留 existence supervision

在正式训练前必须通过一个混合：

    S+ + S-

的小 batch forward/backward 验证。

确认：

S+：
    localization loss 有效
    DQ-CGP binding/routing相关 loss 有效
    existence loss 有效

S-：
    不计算需要真实 temporal GT 的 localization supervision
    不伪造时间窗
    existence loss 有效


============================================================
十二、阅读顺序
============================================================

首先阅读 GMR_Unseen：

1.
README.md

2.
docs/PROJECT_HANDOFF.md

3.
docs/reports/semantic_existence_multisplit_results.md

4.
experiments/correspondence_generalization/
2026年9月30日_正则化与残差适配的未见动作泛化实验/
INTRODUCTION.md

5.
experiments/correspondence_generalization/
2026年9月30日_存在与定位共同泛化的机制诊断/
EXPERIMENT_RECORD.md

6.
experiments/correspondence_generalization/
2026年9月30日_存在与定位共同泛化的机制诊断/
DECISION.md

7.
experiments/event_binding_generalization/HANDOFF.md

8.
experiments/event_binding_generalization/report/REPORT.md

9.
experiments/event_binding_generalization/report/DECISION.md


然后阅读 Falshvtg-gmr-DQ-CGP：

1.
README.md

2.
models/flashvtg_dq_cgp_v3_gmr/README.md

3.
models/flashvtg_dq_cgp_v3_gmr/ANTI_COLLAPSE_DESIGN.md

4.
models/flashvtg_dq_cgp_v3_gmr/model_config.py

5.
models/flashvtg_dq_cgp_v3_gmr/dq_cgp.py

6.
models/flashvtg_dq_cgp_v3_gmr/model.py

7.
models/flashvtg_dq_cgp_v3_gmr/run.py

8.
training/flash_vtg_gmr/dataset.py

9.
training/flash_vtg_gmr/train.py

10.
training/flash_vtg_gmr/inference.py

11.
scripts/train_flashvtg_dq_cgp_v3.sh


============================================================
十三、阅读后不要立刻训练
============================================================

先向我输出一份 compatibility audit。

必须回答：

A.
DQ-CGP v3 是否可以在不修改核心模型的情况下，
直接使用 semantic_existence_v2 数据训练？

B.
数据 JSONL 字段是否兼容？

特别检查：

    qid
    vid
    query
    duration
    relevant_windows

C.
empty relevant_windows 是否被正确保留？

D.
exist_label 是否正确从 empty/non-empty windows 构造？

E.
negative sample 是否只参与 existence supervision，
不会错误进入 localization / DQ-CGP GT loss？

F.
Charades 使用的 CLIP + SlowFast + CLIP text feature
是否与 DQ-CGP 当前模型输入维度兼容？

G.
有哪些 Soccer-GMR 参数必须改成 Charades semantic-existence 参数？

H.
当前 DQ-CGP v3 是否默认：

    dq_cgp_refine_exist=False

如果不是，指出真实状态。

I.
现有 evaluation 是否已经能直接得到：

    seen AUROC
    unseen AUROC
    AUROC gap
    U+ FRR
    U- RR
    raw R1@0.5
    gated R1@0.5
    PairAcc

如果不能，优先复用：

    GMR_Unseen/scripts/analyze_semantic_existence.py

不要重新发明评价定义。


============================================================
十四、实现原则
============================================================

优先采用：

    不修改核心 DQ-CGP 模型
    +
    新增 semantic-existence experiment wrapper/scripts

例如：

    scripts/train_dq_cgp_semantic_existence.sh
    scripts/infer_dq_cgp_semantic_existence.sh

而不是修改原 Soccer-GMR 实验使其失去可复现性。

如果两个仓库需要放在同一个机器目录中，
允许通过环境变量指定：

    GMR_UNSEEN_ROOT
    DQ_CGP_ROOT
    DATA_ROOT
    FEATURE_ROOT
    VIDEO_ROOT
    TEXT_FEAT_DIR

不要硬编码我的绝对路径。


============================================================
十五、第一轮实验
============================================================

先做：

    A1

原因：

FlashVTG-GMR baseline：

    seen AUROC   = 0.8158
    unseen AUROC = 0.5065
    gap          = 0.3093

这是五个 split 中退化最大的一个。

第一轮：

    原始 frozen DQ-CGP v3
    +
    A1
    +
    matched FlashVTG training protocol
    +
    100 epochs
    +
    seed 3407

第一轮不允许根据 U+ / U- 调参数。


============================================================
十六、训练前必须做的 sanity checks
============================================================

正式 100 epoch 前完成：

1. 数据统计

确认 A1：

    train
    val_seen
    test

样本数量和 quadrant 数量与 release 一致。

2. Feature coverage

检查所有使用样本的：

    SlowFast
    CLIP video
    CLIP text

是否完整。

不能默默 skip 大量样本。

3. 一个 positive sample forward。

4. 一个 negative sample forward。

5. mixed S+/S- batch forward/backward。

6. 确认所有 loss finite。

7. 确认 negative 没有假的 localization GT。

8. 确认 pred_exist_score 正常输出。

9. 确认 test inference 覆盖所有 qid。

10. 最好验证：

    dq_cgp_beta = 0

时 DQ-CGP candidate refinement 退化为 baseline path。

这个只作为 implementation sanity check，
不要把它当正式主实验结果。


============================================================
十七、checkpoint selection
============================================================

必须遵守原 benchmark protocol。

不能看 U+/U- 来挑 checkpoint。

checkpoint selection 只能用：

    seen validation

如果 GMR_Unseen 原来的 FlashVTG semantic experiment
已经有固定的 checkpoint selection 规则，
请严格沿用。

不要因为 DQ-CGP 在 Soccer-GMR 使用
MR-full-mAP selection，
就未经检查直接套到这个实验。

先确认 GMR_Unseen 实际训练代码中的 selection rule，
然后 matched。


============================================================
十八、最终 A1 输出
============================================================

最重要的是给我以下对比：

                       Flash baseline      DQ-CGP       Delta
Seen AUROC
Unseen AUROC
Gap = Seen-Unseen
Gap recovery
U+ raw R1@0.5
U+ gated R1@0.5
U+ FRR
U- RR
PairAcc

其中主分析只需要回答：

    1. unseen AUROC 有没有提高？
    2. seen-unseen gap 有没有缩小？
    3. gap 缩小是不是因为 unseen 变好，而不是 seen 崩掉？

请明确分类：

CASE A:

    unseen AUROC ↑
    gap ↓
    seen 基本保持

结论：

    有证据表明 DQ-CGP 在 A1 上缓解了
    FlashVTG-GMR 的 seen→unseen degradation。

CASE B:

    gap ↓
    但主要因为 seen AUROC ↓

结论：

    不能认为是有效的泛化恢复。

CASE C:

    unseen AUROC 没明显改善
    gap 没缩小

结论：

    当前 frozen DQ-CGP 没有缓解 A1 degradation。

CASE D:

    localization 明显改善
    但 unseen existence AUROC 不改善

结论：

    DQ-CGP 可能改善 unseen localization，
    但没有缓解本项目最核心的 existence generalization degradation。

本实验最关心 CASE A。


============================================================
十九、A1 完成以后
============================================================

如果 frozen DQ-CGP v3 在 A1 上存在可信的正向 signal，

下一步不要继续针对 A1 调参。

直接使用同一套冻结参数运行：

    A2_alt
    A3
    C1
    C2_alt

最终形成：

Split      baseline gap     DQ-CGP gap     gap recovery
A1
A2_alt
A3
C1
C2_alt

同时报告每个 split：

    baseline unseen AUROC
    DQ unseen AUROC
    Δ unseen AUROC

这样才能判断缓解是否重复出现。


============================================================
二十、不要过度解释结果
============================================================

如果 DQ-CGP 成功：

可以说：

    在这些冻结的 Charades semantic-existence splits 上，
    DQ-CGP 减缓了 FlashVTG-GMR 的 seen→unseen degradation。

不能自动说：

    event binding 是退化的真正原因。

因为之前 event_binding_generalization 的结果是：

    INCONCLUSIVE

如果 DQ-CGP 失败：

也不能说：

    event binding 不重要。

只能说：

    当前这个 DQ-CGP v3 实现没有成功缓解该 benchmark 上的 degradation。


============================================================
二十一、本次任务不要做的事情
============================================================

除非我之后明确要求，否则不要：

- 设计 DQ-CGP v4；
- 引入新的 backbone；
- 添加新的大模型；
- 修改数据划分；
- 修改 U 定义；
- 根据 test 调参数；
- 为追求指标更改 U+ / U- 样本；
- 用 U test 选 checkpoint；
- 使用 unseen AUROC 做 early stopping；
- 把 rejection rate 当主成功标准；
- 把 calibration 当解决 AUROC degradation 的方法；
- 展开新的机制研究；
- 自动进行完整五 split 大规模 sweep。

============================================================
二十二、你现在应该执行的任务
============================================================

现在先不要训练。

按照上述顺序阅读两个仓库。

然后只给我：

1. 你对本次实验目标的复述；
2. 两个仓库的 compatibility audit；
3. 需要新增或修改的最小文件列表；
4. A1 的精确训练命令；
5. A1 的精确 inference 命令；
6. 四象限 evaluation 命令；
7. 正式训练前 sanity-check 计划；
8. 是否存在任何会导致实验与原 FlashVTG baseline 不 matched 的问题。

只有确认这些内容后，再开始代码修改和训练。

始终记住本次实验的唯一核心问题：

    DQ-CGP 能不能缓解我们在这个新 semantic-existence 数据集上
    观察到的 FlashVTG-GMR seen → unseen degradation？

不要把任务扩展成别的问题。