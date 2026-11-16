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

同一输出目录同时只允许一个写入进程。通过脚本重复执行正在运行的 BREC、`triple-encodings` 或 `triple-ppgn` 时，会显示已有调度 PID 和最近进度，不再启动工作进程。带 `--resume` 的重复请求正常返回；未带时返回状态 2。`--resume` 真正续跑需等原任务中断并释放写锁；不要删除 `.writer.lock` 或另开相同目录的任务。

`tools/run_rebuttal.sh` 默认显示进度条，包括已处理数量、百分比、已用时间及预计剩余时间；尚无完成项时 ETA 显示为未知。BREC 续跑会计入已有记录，长条件计算期间仍持续刷新已用时间。终端中原地更新，重定向日志时保留原输出并每 30 秒写一条进度。机制对照和汇总按整项任务显示，其他任务分别按条件、图、epoch 或预测任务计数；已处理数量包含未确定和资源受限记录，认证结论以结果汇总为准。设置 `REBUTTAL_PROGRESS=0` 可关闭显示。进度包装层不改变实验计算代码或运行身份。

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

官方 reliability 对照使用已安装的 nauty 求规范顶点顺序，再独立核验映射后的整数邻接矩阵完全相等，避免 NetworkX VF2 在高度对称图上的长时间回溯搜索。启动工作进程前，对本次请求的全部 reliability 图对预检，失败时报告 pair ID 并停止启动。一条条件需完成主判别、16 次主图重标号和全部 reliability 对照才会写入结果；因此忙碌的工作进程不一定已产生新记录。

默认按可用 CPU 数并行运行独立条件，每个进程的数值库保持单线程；CPU 绑定复用原 `brec/schedule.py` 的物理核优先顺序。内存 `memory_gib: 52` 是调度器与所有工作进程的 RSS 总预算，调度器按任务预估内存准入并监控实际 RSS，同时保留默认 8 GiB 系统可用内存；共享页可能重复计入 RSS，统计偏保守。超出总预算时停止占用最大的未完成条件并记录 `resource_limited / scheduler_memory_limit`，不自动改变数值路径。结果逐条件原子保存，中断会清理工作进程，`--resume` 跳过已有记录并继续未完成条件。

不设每条件时间上限（`seconds_per_condition: null`），兼容匹配边上限仍为 200 万。需要限时时，将 `seconds_per_condition` 设为正数秒数。资源设置应在正式结果前按服务器能力固定；修改代码、配置或并发设置后使用新输出目录，已启动的进程继续使用启动时读取的设置。

调度器每 30 秒输出 `BREC heartbeat`，包括活动进程数、距上次结果的时间、总 RSS、各计算阶段及停留最久的条件。`progress.json` 同步保存所有活动条件的 PID、绑定 CPU、RSS、当前阶段、阶段耗时及精度／置换编号；该文件是运行状态，不作为完成记录。阶段包括 `main_features`、`main_readout`、`main_relabel`、`reliability_mapping`、`reliability_features` 和 `reliability_relabel`。心跳由父进程独立更新，即使某个工作进程长时间不离开当前阶段，也能看到停留时间增长；不会自动超时或改变计算路径。

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
- `progress.json`：每 30 秒更新的工作进程状态；结束或中断时保存最终调度状态。

证据摘要用于解释结果，不是脱离输入即可独立验证的完整证书。上述 `brec` 命令覆盖二元 BREC；完整三元／joint 与神经输出使用下文的独立入口，`mechanisms` 的有限三元结果不能代替这些范围。

所有 BREC 启动入口继续固定随机种子、数值库单线程及 PyTorch 严格确定性算法。历史神经 BREC 保留固定物理 batch、dtype 和 GPU 分配，不再按 OOM 自动切换计算路径。相同输入／配置／软件／硬件下的重复运行与重标号检查分别报告。

如需重复运行确认，以下命令会运行两次完整 BREC，再直接比较结果记录和证据摘要：

```bash
bash tools/run_rebuttal.sh brec-reproduce
# 或比较已经分别完成的两份 v2 BREC 输出：
python -m brec.reproducibility --first outputs/brec_run1 --second outputs/brec_run2 --out outputs/brec_repeat.json
```

比对排除耗时；未完成或资源受限不能通过。`passed: true` 只表示所请求的二元 BREC 条件记录一致，不核查神经权重、不承诺跨平台一致，也不是任意重标号的数学证明。

## 先运行三元编码本身

`triple-encodings` 不构造或训练神经网络。按 Q=1→2→4 分阶段遍历全部 BREC 400 对图，比较三元 `baseline`（polynomial＋endpoint）与 `joint`，两种基，以及完整多重集和累积均值矩 P≤1/2/4/8/16，共 72 个设置、28800 条条件记录。完整字典的宽度分别为：

| Q | baseline | joint |
|---|---:|---:|
| 1 | 19 | 27 |
| 2 | 37 | 64 |
| 4 | 91 | 216 |

计算使用有理系数与无平方因子根式的精确表示，没有浮点容差。采用随机游走矩阵与对称归一化的相似变换，将多项式矩阵幂化为缩放整数矩阵；endpoint 和 joint 的值在同一精确数域内构造。joint 仅在整数绝对值上界证明所有乘积与部分和均能由 int64 精确表示时批量计算，否则使用 Python 大整数。所有原始有序三元组，包括重复顶点，均计入分母 n³。

为避免存储 n³×通道数的大表，先用图内自同构的稳定子链取得三元组轨道代表及其重数。轨道只压缩求值，不作为编码通道；完整多重集按轨道重数精确比较。小图测试独立核对压缩后的加权多重集与逐三元组完整枚举。通道按列流式求值，已找到的差异可由通道子集认证；均值的一阶和使用精确求和恒等式，其余阶数使用加权通道直方图。

每侧保留 8 次固定重标号、官方 reliability 对照及 2 次额外置换。每次控制检查全部多项式 bank 条目的精确对齐，并抽取 16 个三元组检查完整派生特征行的精确对齐；这是采样行检查，不声称重算了每次重标号后的完整三元特征表。每对图、每 Q 的两个基和全部读出共享同一批控制检查，全部通过后才写出条件记录。

```bash
source ~/miniforge3/bin/activate gnn270
export BREC_FILE=/home/chenrq/GSE_Toward_kFWL/data/brec_raw/brec_v3.npy
export REBUTTAL_OUT=outputs/rebuttal_triples_q124
bash tools/run_rebuttal.sh triple-encodings --workers 28
# 同一来源、配置和输入下继续：
bash tools/run_rebuttal.sh triple-encodings --workers 28 --resume
```

总 RSS 预算 52 GiB、系统保留 8 GiB、不设条件超时。`records/` 保存精确判别、数值证据和控制范围，`jobs/` 保存每对图／Q 的调度状态，`orbits/` 保存绑定输入与运行身份的轨道缓存；`summary.json` 保留每设置 400 的分母，`progress.json` 显示活动任务的计算阶段。不会自动接续 `triple-ppgn`。

## 补齐二元 P=8 与三元 Q=8/16

新增两个独立配置，旧配置与结果继续保留：

| 编码 | 新增 Q | 新增 P | 条件记录 |
|---|---|---|---:|
| k=2 | 2、4、8、16 | 累积 P≤8 | 6400 |
| k=3 | 8、16 | 累积 P≤1/2/4/8，另含完整多重集 | 16000 |

`brec_p8.json` 的 `readouts: ["mean"]` 跳过已完成的二元完整多重集，只补均值矩。二元继续使用原来的精确分数／128→256→512 bit 区间协议和全部重标号控制。`triple_encodings_q8_q16.json` 沿用精确三元协议与全部控制；joint 字典在 Q=8/16 时有 1000/5832 个通道。

以下入口先运行二元补充，完成后自动运行三元 Q=8→16；每阶段内部最多 28 个 CPU 工作进程，共用阶段性的 52 GiB 总 RSS 预算和 8 GiB 系统保留，不设条件超时。阶段失败或仍有已有写入任务时，不会继续启动下一阶段。

```bash
source ~/miniforge3/bin/activate gnn270
export BREC_FILE=/home/chenrq/GSE_Toward_kFWL/data/brec_raw/brec_v3.npy
export REBUTTAL_OUT=outputs/rebuttal_budget_extension
bash tools/run_rebuttal.sh budget-extend --workers 28
# 中断后续跑：
bash tools/run_rebuttal.sh budget-extend --workers 28 --resume
```

监督进程的阶段状态保存在 `extension.json`，二元结果在 `brec/`，三元结果在 `triple_encodings/`。也可分别使用原入口加 `--config configs/rebuttal/brec_p8.json` 或 `--config configs/rebuttal/triple_encodings_q8_q16.json`。变更协议使用新目录，避免与旧结果混写。

补充配置使用 `memory_reservation_scale: 0.6` 校准启动时的静态内存预估，以利用实测余量增加并发；实际 RSS 总预算仍为 52 GiB，系统仍保留 8 GiB。调度器持续按实测 RSS 提高进程预留量，并保留超预算监控。未指定时系数为 1，旧协议的默认准入行为不变。`progress.json` 额外记录预留字节数、校准系数及可用 CPU 槽位。

## 三元 PPGN 配对 BREC

`triple-ppgn` 借用 `~/GSE_Toward_kFWL/` 的旧版三元 PPGN：4 个 block、宽度 16、每分支两层通道 MLP、共享 pivot 收缩、有符号立方根以及 5 种相等模式的均值读出。它与当前 `brec/ppgn_models.py` 的另一版架构不同；独立的 `legacy_triple_ppgn.py` 保留旧骨干并记录来源校验和。

完整设置为 400 对图 × 3 个配对 seed × 3 组输入，共 3600 个训练条件。原输入保留旧版 Chebyshev Q=8 的 61 个通道；新增编码采用 Q=1，包含 19 个三元 baseline 通道及 8 个 joint 通道。三组统一使用 88 个输入槽：

| 组 | 前 61 槽 | 后 27 槽 |
|---|---|---|
| `original` | 旧版输入 | 27 个零槽 |
| `baseline` | 旧版输入 | 19 个三元 baseline 通道＋8 个零槽 |
| `joint` | 旧版输入 | 19 个三元 baseline 通道＋8 个 joint 通道 |

Q=1 baseline 通道已包含在旧输入中，是保留信息的冗余对照。`baseline` 与 `joint` 只相差最后 8 槽；三组模型形状、参数量、配对初始化、图顺序、优化器、最大 20 个 epoch 和早停规则相同。所有有序三元组均保留，包括重复顶点。新增 K111 只对实际共同邻居累加，避免构造 pivot×全部三元组的四阶数组。

计算固定为 float32、关闭 TF32、单数值线程，不按 OOM 自动切换 batch 或 dtype。每次只保留一张图的反向激活，先计算逻辑 batch 的 embedding 梯度，再逐图重计算并反传；该骨干无 BatchNorm 或 Dropout，链式法则给出与逻辑 batch 损失相同的梯度，测试独立验证参数更新一致性。有效 batch 仍为 32 张图，另启用 block checkpoint。实际 float32 累加顺序与旧物理 batch 运行不同，属于这次独立冻结的计算协议。

每个条件使用官方 32 对训练重标号和 32 对 reliability 重标号。保存模型权重、两组各 64 个图的 embeddings、逐 epoch 损失、RPC 协方差／SVD 检查及显存峰值；数值不可靠或违反参考上界的条目保持未确定，不当作未区分。汇总同时给出每 seed 的分母 400 计数及三组逐图配对的 gained/lost/both/neither。

```bash
source ~/miniforge3/bin/activate gnn270
export BREC_FILE=/home/chenrq/GSE_Toward_kFWL/data/brec_raw/brec_v3.npy
export REBUTTAL_OUT=outputs/rebuttal_triple_ppgn_q1
bash tools/run_rebuttal.sh triple-ppgn --devices cuda:0 cuda:1
# 中断后，使用同一配置与输入续跑：
bash tools/run_rebuttal.sh triple-ppgn --devices cuda:0 cuda:1 --resume
```

两张 GPU 各一个工作进程；总 CPU RSS 预算 52 GiB、系统保留 8 GiB、不设每条件时间上限。每 30 秒更新 `progress.json` 和当前汇总；每个 pair/seed 的日志在 `logs/`，工作进程阶段在 `workers/`。启动时冻结 Python 源码供全部子进程使用。失败即停止整个调度并保留已完成条件，避免悄悄改变实验设置。

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

测试覆盖区间运算的精确包络、独立分数公式、两种基与全部默认 Q 预算、累积均值矩、三元提升、重复行匹配、非自逆重标号、官方 reliability 映射与非同构拒绝、禁止退回 VF2、并行与重复运行一致性、工作进程阶段可见性、内存限制、中断清理、原子保存及续跑身份。配置错误在启动工作进程前拒绝；复现检查同时核对条件覆盖、记录身份和汇总一致性。这些测试使用小图，不替代正式 BREC 全部条件的运行与结果审查。
