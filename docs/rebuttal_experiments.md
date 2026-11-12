# Rebuttal 补充实验：服务器运行说明

协议 `rebuttal-2026-11-12-v2`。代码按实验计算、运行流程和结果汇总组织；没有捆绑旧实验结果。`mechanisms`已在服务器完成：精确base／joint多重集比较及换基第三矩差1/331776与理论一致；完整BREC、对象域精度／容差和蛋白质预测仍待执行。

## 环境与输入

在 Linux、Python 3.12 下，从本仓库根目录运行：

```bash
python -m venv .venv-rebuttal
source .venv-rebuttal/bin/activate
python -m pip install -r requirements/rebuttal.txt
# 蛋白质实验另需服务器对应的 PyTorch，再安装：
python -m pip install -r requirements/rebuttal-protein.txt

export REBUTTAL_OUT=outputs/rebuttal
export BREC_FILE=/server/data/BREC/brec_v3.npy
export MUTAG_TOPOLOGY=/server/data/MUTAG/topology.npz
export TU_DATA_ROOT=/server/data/tu
```

将输入路径换成真实位置。BREC 使用官方 51200 条 graph6 数组，SHA-256 固定在配置中。MUTAG 使用包含 `edges/counts/offsets` 的 188 图 NPZ。蛋白质接受本地 TU raw 文本或 ZIP；仅显式 `--download` 才下载数据。

每项默认写入新目录，拒绝覆盖。`--resume` 只接受相同代码、配置和输入；修改代码或配置后使用新目录。

## 实验与命令

| 实验 | 命令 | 回答的问题 |
|---|---|---|
| 1. joint 与均值矩 | `mechanisms` | rook／Shrikhande 的原始区分与读出区分是否不同 |
| 2. 换基与读出 | 同一个 `mechanisms` | Q=1、P=3 下，保留原始信息的换基能否改善均值矩 |
| 3. BREC | `brec` | 用精确／区间计算确认二元编码的区分，保留未确定项 |
| 4. 对象域 | `objects` | MUTAG 三域 d=14 的结果如何受精度、容差和单位影响 |
| 5. 蛋白质预测 | `protein-prepare/train/predict/report` | 相同输入宽度和训练预算下，joint 是否改变预测表现 |

分别运行所需实验，不自动启动全部任务：

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec
bash tools/run_rebuttal.sh objects
```

## BREC 的计算与结果

`configs/rebuttal/brec.json` 固定两个基、polynomial／endpoint、Q=2/4/8/16，以及完整多重集和累积均值矩 P=1/2/4，共 64 个二元设置。两图都正则时使用精确分数；其他条件从整数邻接矩阵生成 128→256→512 bit 区间。区间后端使用任意精度整数端点，不依赖 Arb，也不包裹已舍入的浮点特征。

均值差区间排除 0 才报告区分；完整多重集用投影计数或兼容匹配缺口确认。区间重叠或存在兼容完美匹配不能证明相等，记录为未确定。每侧保留 8 次固定重标号、官方 reliability 对照及 2 次额外置换；图同构仅用于对齐已知同构对照，不参与主图对编码。

默认按可用 CPU 数并行运行独立条件，每个进程的数值库保持单线程；CPU 绑定复用原 `brec/schedule.py` 的物理核优先顺序。内存 `memory_gib: 52` 是调度器与所有工作进程的 RSS 总预算，调度器按任务预估内存准入并监控实际 RSS，同时保留默认 8 GiB 系统可用内存；共享页可能重复计入 RSS，统计偏保守。超出总预算时停止占用最大的未完成条件并记录 `resource_limited / scheduler_memory_limit`，不自动改变数值路径。结果逐条件原子保存，中断会清理工作进程，`--resume` 跳过已有记录并继续未完成条件。

不设每条件时间上限（`seconds_per_condition: null`），兼容匹配边上限仍为 200 万。需要限时时，将 `seconds_per_condition` 设为正数秒数。资源设置应在正式结果前按服务器能力固定；修改代码、配置或并发设置后使用新输出目录，已启动的进程继续使用启动时读取的设置。

例如限制最多 28 个进程，总内存预算仍为 52 GiB：

```bash
export REBUTTAL_OUT=outputs/rebuttal_parallel
bash tools/run_rebuttal.sh brec --workers 28
# 同一代码、配置和并发设置下续跑：
bash tools/run_rebuttal.sh brec --workers 28 --resume
```

也可设置 `BREC_WORKERS=28`，适用于 `brec` 和 `brec-reproduce`。`--reserve-gib` 调整系统可用内存保留量。输出为：

- `records/*.json`：设置、判别、精度尝试、重标号结果及 `evidence` 数值摘要。摘要包含区分通道与差值、计数界或匹配缺口，不再重复导出整份特征见证。
- `summary.json`：每设置分母 400，分别列已确认区分、已确认相等、未确定、资源不足和未执行项。区分下界较大不自动证明真实表达力更强。
- `manifest.json`：输入／代码／配置／环境身份及运行状态。

证据摘要用于解释结果，不是脱离输入即可独立验证的完整证书。新程序仍只覆盖二元 BREC，不覆盖完整三元／joint 或神经输出；`mechanisms` 的有限三元结果不能代替这些范围。

所有 BREC 启动入口继续固定随机种子、数值库单线程及 PyTorch 严格确定性算法。历史神经 BREC 保留固定物理 batch、dtype 和 GPU 分配，不再按 OOM 自动切换计算路径。相同输入／配置／软件／硬件下的重复运行与重标号检查分别报告。

如需重复运行确认，以下命令会运行两次完整 BREC，再直接比较结果记录和证据摘要：

```bash
bash tools/run_rebuttal.sh brec-reproduce
# 或比较已经分别完成的两份 v2 BREC 输出：
python -m brec.reproducibility --first outputs/brec_run1 --second outputs/brec_run2 --out outputs/brec_repeat.json
```

比对排除耗时；未完成或资源受限不能通过。`passed: true` 只表示所请求的二元 BREC 条件记录一致，不核查神经权重、不承诺跨平台一致，也不是任意重标号的数学证明。

## 对象域结果

`configs/rebuttal/objects.json` 固定 MUTAG188、node/edge/pair、d=14、两种精度和预设容差／单位网格。`arrays/*.npz` 保留特征、掩码、置换和轨道；`records/*.json` 保留 RC、MIS 最优状态与边界参考；`summary.json` 同时报告共同有效队列及未确定项。数值相似图的精确 MIS 不等于精确结构特征，本实验不训练模型。

## 蛋白质预测

默认 ENZYMES，可在查看测试结果前改为 PROTEINS 或 PROTEINS_full：

```bash
export PROTEIN_DATASET=ENZYMES
export PROTEIN_DEVICE=cuda:0
bash tools/run_rebuttal.sh protein-prepare
bash tools/run_rebuttal.sh protein-train
bash tools/run_rebuttal.sh protein-predict
bash tools/run_rebuttal.sh protein-report
```

无 GPU 时设 `PROTEIN_DEVICE=cpu`。需下载时，显式使用 `python -m rebuttal_experiments protein prepare --dataset ENZYMES --data-root "$TU_DATA_ROOT" --download --out "$REBUTTAL_OUT/protein_inputs"`。

三组共同使用全有序三元组和 27 槽模型输入：base 为 19 通道加 8 个零槽；joint 加全部 8 个 Q=1 joint 通道；nonjoint 加 6 个 M² 提升和 h₁(u)、h₁(v)。nonjoint 使用 degree-2 seeds，等宽不等于同 Q 或同构造成本。原生节点／边属性保留；完整行先经 MLP 交互，再聚合。

5 个外层 fold、3 个配对 seed，共 45 个任务。保持初始化、图顺序、参数量和更新预算配对；只按未舍入验证 NLL 选 checkpoint，并列选较早 epoch，训练不评估 test。汇总保留逐图 logits、每 seed 的 OOF accuracy／NLL、配对差及成本。不能据预测收益证明需要 3-FWL；记录分层不等于控制了蛋白质同源关系。

准备阶段直接生成实际数据特征，不再自动重跑理论机制自检。保留原始输入、配置、代码及划分身份，不再为全部特征缓存生成校验和，也不在每次训练／预测启动时重读全部缓存做校验。图超过三元缓存或资源上限时明确中止，不静默删图。`--folds/--seeds/--variants` 可拆分训练，分别写入不同目录；`report --runs` 可合并它们。`--resume` 跳过相同身份的完成任务；显式 `--retry-incomplete` 会删除本运行中未完成任务的目录，再从头训练，不额外归档。

## 代码阅读顺序

- BREC：`brec.py` 组织运行，`scheduling.py` 并行调度，`encodings.py` 生成通道，`readouts.py` 作判别。
- 对象域：`objects.py` 组织控制和汇总，`object_comparison.py` 保存数值计算。
- 蛋白质：`protein.py` 只解析命令；`protein_data.py` 准备输入，`protein_training.py` 训练，`protein_results.py` 预测与汇总，`protein_model.py` 定义模型。

每次运行的 `manifest.json` 记录配置、输入、代码和环境身份。仓库不再保留旧结果表、静态源码清单或归档导出工具。

## BREC 代码验证

```bash
python -m unittest discover -s tests -p 'test_brec_*.py' -v
```

测试覆盖区间运算的精确包络、独立分数公式、两种基与全部默认 Q 预算、累积均值矩、三元提升、重复行匹配、非自逆重标号、官方 reliability 映射、并行与重复运行一致性、内存限制、中断清理、原子保存及续跑身份。配置错误在启动工作进程前拒绝；复现检查同时核对条件覆盖、记录身份和汇总一致性。这些测试使用小图，不替代正式 BREC 全部条件的运行与结果审查。
