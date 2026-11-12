# Rebuttal 补充实验：服务器运行说明

协议 `rebuttal-2026-11-12-v2`。代码按实验计算、运行流程和结果汇总组织；没有捆绑旧实验结果。`mechanisms`已在服务器完成：精确base／joint多重集比较及换基第三矩差1/331776与理论一致；完整BREC和蛋白质预测仍待执行。

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
export TU_DATA_ROOT=/server/data/tu
```

将输入路径换成真实位置。BREC 使用官方 51200 条 graph6 数组，SHA-256 固定在配置中。蛋白质接受本地 TU raw 文本或 ZIP；仅显式 `--download` 才下载数据。

每项默认写入新目录，拒绝覆盖。`--resume` 核对数据、配置、核心计算代码和环境，并保留已完成记录；仅删除停用的补充模块或整理运行入口不要求重算。原记录身份不改写，当前源码清单另记在运行manifest中。

## 实验与命令

| 实验 | 命令 | 回答的问题 |
|---|---|---|
| 1. joint 与均值矩 | `mechanisms` | rook／Shrikhande 的原始区分与读出区分是否不同 |
| 2. 换基与读出 | 同一个 `mechanisms` | Q=1、P=3 下，保留原始信息的换基能否改善均值矩 |
| 3. BREC | `brec` | 用精确／区间计算确认二元编码的区分，保留未确定项 |
| 4. 蛋白质预测 | `protein-prepare/train/predict/report` | 相同输入宽度和训练预算下，joint 是否改变预测表现 |

分别运行所需实验，不自动启动全部任务：

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec
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

## 蛋白质预测

固定运行 **PROTEINS 和 ENZYMES**。在 100 GiB 内存、32 GiB 显存的服务器上依次执行，避免两个训练进程争用同一张 GPU：

```bash
export TU_DATA_ROOT=/path/to/tu
export PROTEIN_DEVICE=cuda:0
bash tools/run_rebuttal.sh protein-both
# 数据尚未放到 TU_DATA_ROOT 时，改用：
# bash tools/run_rebuttal.sh protein-both --download
```

两套输出分别在 `$REBUTTAL_OUT/protein_encodings/PROTEINS/` 与 `$REBUTTAL_OUT/protein_encodings/ENZYMES/`，各含 `protein_inputs/`、`protein_training/`、`protein_report.json`。未设置 `REBUTTAL_OUT` 时使用 `outputs/rebuttal`。缓存目录应位于空间充足的本地高速磁盘。旧27槽实验保留在旧目录，不能将旧checkpoint混入新64槽协议；脚本使用新目录避免覆盖。

也可分阶段运行某一个数据集：

```bash
export PROTEIN_DATASET=PROTEINS  # 或 ENZYMES
bash tools/run_rebuttal.sh protein-prepare
bash tools/run_rebuttal.sh protein-train
bash tools/run_rebuttal.sh protein-predict
bash tools/run_rebuttal.sh protein-report
```

无 GPU 时设 `PROTEIN_DEVICE=cpu`。`PROTEIN_CONFIG`可指定共同配置文件，默认`configs/rebuttal/protein.json`；单数据集可用`PROTEIN_OUT`指定新输出位置，`protein-both`始终按数据集隔离目录。

九组均使用全部有序三元组（含重复顶点）。选定字典完整保留后补零至64个结构槽，不截断或随机投影；相同backbone给出三个节点的表示，与结构行拼接后送入共同MLP，再对全部三元组平均。这样可以学习结构与节点属性的关联。所有组的模型参数量相同；`none`只将结构槽置零，仍使用相同的节点表示和读出。

| 设置 | 补零前通道数 | 目的 |
|---|---:|---|
| `none` | 0 | 共同模型中不加入结构编码的基线 |
| `polynomial_q1` | 19 | 常数及I、A、M的六个有序坐标提升 |
| `nonjoint_q1` | 27 | 上述19通道＋六个M²提升及h₁(u)、h₁(v)，保留先前nonjoint对照 |
| `joint_q1` | 27 | 上述19通道＋全部8个Q=1 joint通道 |
| `joint_q1_shifted` | 27 | 完整换基(I,M)→(I,I+M)，不增加原始信息 |
| `polynomial_q2` | 25 | 保留Q=1通道，再加入六个M²坐标提升 |
| `endpoint_q2` | 37 | 上述25通道＋原定义的全部12个Q=2 endpoint通道 |
| `joint_q2` | 64 | 上述37通道＋全部27个Q=2 joint通道 |
| `joint_q2_chebyshev` | 64 | 完整换基(I,M,M²)→(I,M,2M²−I)，不增加原始信息 |

`nonjoint_q1`的名称指以Q=1基础通道作增广，其额外通道实际使用degree 2。joint来自3-FWL代数，包含共享pivot收缩K_{abc}(u,v,w)=Σₜ B_a(u,t)B_b(v,t)B_c(w,t)；有限Q=1/2字典不等同于完整3-FWL。

主比较预先固定为 **`joint_q2 − endpoint_q2` 的accuracy差**，同时报告NLL差。前者保留后者全部通道再增广joint。还预设degree增加、Q=1 joint／nonjoint、joint Q=1→2及两组完整换基等比较。所有低预算设置均可从完整joint Q=2字典恢复；这保证原始信息不丢失，不保证训练或泛化一定改善。换基对照只检验相同信息是否更易被模型利用。

每个数据集5个外层fold、3个配对seed、9个变体，各135个任务，合计 **270个任务**，每任务100 epochs、batch size 4。保持初始化、图顺序、参数量和更新预算配对；checkpoint按未舍入验证NLL选择，并列取较早epoch。两个数据集分别报告全部预设配置的逐图logits、每seed OOF accuracy／NLL、10项预设配对差和成本，保留零／负结果。另在每fold／seed内按验证NLL选择编码，并独立汇总其外层测试表现；并列按配置中的固定顺序取首项，绝不按测试分数选。fold×seed不作为独立数据集重复，也不声称已控制蛋白质同源关系。

全部设置共享一次准备：按16384行写入float64的25个polynomial和27个joint通道，endpoint按需生成，两组换基从同一缓存派生，不各存一份全三元组。joint按固定u的平面计算，含identity的项直接化简，其余只对相应种子行的非零位置求和。所需特征磁盘为每图 **416n³字节**，加文件头和小规模统计；准备时检查实际空间，并另留5 GiB。准备／训练进程内存上限均为92 GiB，在100 GiB服务器上保留8 GiB余量；这是检查上限，不是实测占用或显存上限。分块仍处理全部n³对象，不能据此承诺运行时间。

训练每次读取最多65536行。checkpoint保存分块加载函数，反向传播时重读同一块；节点表示显式作为checkpoint参数，保留其梯度及对应设备的dropout随机状态，使用`use_reentrant=False`。标准化只合并训练图的通道和及Gram矩阵，换基后的一、二阶统计直接由其推得；方差使用相对二阶矩的数值下限，避免放大近常数通道的舍入噪声。无需为每个任务扫描所有缓存。默认不设准备／训练时间截止，不删图或采样。

准备阶段不重跑已完成的理论机制实验。保留原始输入、配置、代码及划分身份，不重复校验全部特征文件。启动脚本自动使用 `--resume`，跳过相同身份的完成图／训练任务；预测跳过已有匹配结果。中断的单个训练任务须检查后显式用 `protein-train --retry-incomplete` 从头重跑，不额外归档。`--folds/--seeds/--variants` 可拆分训练，分别写入不同目录；`report --runs` 可合并同一数据集的任务。只改蛋白质代码不会使已完成的 BREC 条件失效；BREC 自身的输入、数值代码、配置和环境仍须匹配。

## 代码阅读顺序

- BREC：`brec.py` 组织运行，`scheduling.py` 并行调度，`encodings.py` 生成通道，`readouts.py` 作判别。
- 蛋白质：`protein.py`解析命令，`protein_encodings.py`定义编码及完整换基，`protein_data.py`准备输入，`protein_training.py`训练，`protein_results.py`预测与汇总，`protein_model.py`定义共同模型。

每次运行的 `manifest.json` 记录配置、输入、代码和环境身份。仓库不再保留旧结果表、静态源码清单或归档导出工具。

## BREC 代码验证

```bash
python -m unittest discover -s tests -p 'test_brec_*.py' -v
```

测试覆盖区间运算的精确包络、独立分数公式、两种基与全部默认 Q 预算、累积均值矩、三元提升、重复行匹配、非自逆重标号、官方 reliability 映射、并行与重复运行一致性、内存限制、中断清理、原子保存及续跑身份。配置错误在启动工作进程前拒绝；复现检查同时核对条件覆盖、记录身份和汇总一致性。这些测试使用小图，不替代正式 BREC 全部条件的运行与结果审查。
