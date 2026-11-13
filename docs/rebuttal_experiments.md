# Rebuttal 补充实验：服务器运行说明

BREC／机制协议 `rebuttal-2026-11-12-v2`；TU使用 `gnnplus-gine-tu-structural-edge-v1`。代码按实验计算、运行流程和结果汇总组织；没有捆绑旧实验结果。`mechanisms`已在服务器完成：精确base／joint多重集比较及换基第三矩差1/331776与理论一致；完整BREC和蛋白质预测仍待执行。

## 环境与输入

BREC／精确机制实验在Linux、Python 3.12环境中，从本仓库根目录运行：

```bash
python -m venv .venv-rebuttal
source .venv-rebuttal/bin/activate
python -m pip install -r requirements/rebuttal.txt

export REBUTTAL_OUT=outputs/rebuttal
export BREC_FILE=/server/data/BREC/brec_v3.npy
```

将输入路径换成真实位置。BREC 使用官方 51200 条 graph6 数组，SHA-256 固定在配置中。TU实验使用下文Co-Sparsify的原始环境和原始PyG loader，不在BREC环境中安装／运行；数据缺失时沿用原loader的下载行为。

每项默认写入新目录，拒绝覆盖。`--resume` 核对数据、配置、核心计算代码和环境，并保留已完成记录；仅删除停用的补充模块或整理运行入口不要求重算。原记录身份不改写，当前源码清单另记在运行manifest中。

## 实验与命令

| 实验 | 命令 | 回答的问题 |
|---|---|---|
| 1. joint 与均值矩 | `mechanisms` | rook／Shrikhande 的原始区分与读出区分是否不同 |
| 2. 换基与读出 | 同一个 `mechanisms` | Q=1、P=3 下，保留原始信息的换基能否改善均值矩 |
| 3. BREC | `brec` | 用精确／区间计算确认二元编码的区分，保留未确定项 |
| 4. 蛋白质预测 | `protein-run/report` | 相同输入宽度和训练预算下，joint 是否改变预测表现 |

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

**Co-Sparsify数据／训练流程 + GNNPlus GINE模型。** 固定Co-Sparsify commit `505f6b50f019dc78edc88653d771667c0c561f5f`，沿用其TU loader、十折划分、Lightning训练、优化器、scheduler、val/acc选模和评估器。`protein_gnnplus.py`将原SPPGN1替换为`gnnplus_official`的`GINEConvLayer`（hidden 256、3层）和`SANGraphHead`（3层、mean pooling、ReLU、dropout 0）；`pair_x`按`(u,v)`映射到原始有向边，`loop_x`拼接到节点输入。两处上游官方文件均不修改。

来源：[运行入口](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/run_tud.py)、[TU配置](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/configs/sppgn/tud.sppgn1.poly.yaml)、[划分](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/datasets/kfold.py)、[训练器](https://github.com/RongqinChen/Co-Sparsify/blob/505f6b50f019dc78edc88653d771667c0c561f5f/lighting_interface/model.py)。

| 项目 | 直接沿用的官方设置 |
|---|---|
| 数据 | `PROTEINS_full`、`ENZYMES`；非cleaned版本。原PyG loader的连续node／edge attributes默认关闭，不额外加常数列或标准化 |
| 划分 | 10折；缺少原10fold文件时StratifiedKFold(shuffle=True, random_state=42)，第r折使用原代码idx_list[r−1]；已有文件直接沿用并记录实际索引 |
| 初始化 | 第r折seed=(r+1)×1000；不额外重复三种seed |
| 网络 | GNNPlus GINE，3层，hidden 256，残差连接、dropout 0、mean pooling及3层SAN图头 |
| 编码宽度 | 官方脚本的8、12、14、16；同一宽度下全部编码模型参数量一致 |
| 训练 | 当前显式覆盖为物理batch 64；训练drop_last=True、AdamW、lr=0.001、weight_decay=0、100 epochs、5 warmup epochs、原cosine scheduler、norm梯度裁剪1.0 |
| 选模 | 原val/acc最大；保留原评估器五位小数与原checkpoint并列处理，不改成NLL |
| 最终评分 | 原脚本将同一个留出折同时作为validation和test；用户明确要求完全沿用。不称为独立test估计 |
| 汇总 | 每个编码／宽度报告10折分数的均值与总体标准差(ddof=0)，沿用原字段test_acc等；保存raw logits与索引 |
| 执行 | Co-Sparsify原GPU Trainer／8个torch线程／4周Timer；模型侧调用GNNPlus原生GINE层和SAN头，并将Comet切换为local_log |

配置中的min_lr=1e-6由原scheduler按原代码解释，不另写一个学习率实现。物理batch及drop_last由原DataLoader处理，不用逐图梯度累积模拟。`configs/rebuttal/protein.json`显式设置`batch_size_override: 64`；该值写入每次manifest和记录。GNNPlus模型不需要SPPGN1专用的`2fwl_connsp`三元索引，因此该变换在此后端下为恒等变换；全部结构编码仍在poly变换的`pair_x`／`loop_x`中生成。

### 九组编码

令宽度K，r=⌊K/2⌋，e=⌊(K−r)/2⌋，j=K−r−e。对K=8／12／14／16，(r,e,j)分别为(4,2,2)、(6,3,3)、(7,3,4)、(8,4,4)。M为对称归一化邻接，B_q=M^q，h_q=B_q 1。

| 设置 | GNNPlus GINE收到的K维编码 |
|---|---|
| `rrwp` | 原生Co-Sparsify RRWP，I,P,…,P^(K−1)，P为行归一化邻接；基线完全走原变换 |
| `mono` | 原生I,M,…,M^(K−1) |
| `cheb` | 原生完整Chebyshev基；与mono同次数空间 |
| `none` | K个零槽；模型和原生节点／边输入不变 |
| `base` | 前r个monomial通道，余下补零 |
| `endpoint` | base＋前e个[h₁(u),h₁(v),h₂(u),h₂(v)]，余下j槽补零 |
| `powers` | endpoint＋前j个已保留B_q(u,v)的逐元素平方；不增加该编码的原始信息 |
| `joint` | endpoint＋J₁,…,Jⱼ，保留endpoint所有槽 |
| `joint_mix` | joint的完整K维正交DCT换基，不截断、不增加原始信息 |

新增的joint点对通道由固定三元读出得到：

\[
J_q(u,v)=A_{uv}\sum_w A_{uw}A_{vw}\sum_t B_q(u,t)B_q(v,t)B_q(w,t).
\]

内层是三元共享pivot构造，外层用三角形条件作固定读出，接入原点对接口，不加可训练模块。无自环图上J₁涉及四个两两相连的不同顶点，保留一种4-clique相关信号；它不是原完整三元特征多重集，也不声称完整3-FWL等价。程序一次只持有一个局部平面，不保存全n³缓存。所有新编码沿用原变换的float32接口；不新增特征标准化。

主比较为同K的`joint − endpoint`，另报告与`powers`、原生`mono`等的预设比较。joint对endpoint有明确保留关系；joint与完整mono在相同K下交换了通道预算，不预设一般支配。`cheb/mono`和`joint_mix/joint`分别控制相同信息的表示方式。全部配置都报告，不再生成“逐折选最好编码后的独立test结果”。

### 服务器运行

使用Co-Sparsify原环境，按其readme的mamba／Sage／PyG及CUDA扩展说明准备；参考环境是Python 3.11、NumPy 1.26、PyTorch 2.2.2/CUDA 12.1。不要将BREC的NumPy 2.4环境依赖装进此环境。适配器无需超出原仓库的新包。

若服务器尚无原仓库，先准备固定版本（已有checkout直接复用）：

```bash
git clone https://github.com/RongqinChen/Co-Sparsify.git Co-Sparsify
git -C Co-Sparsify checkout 505f6b50f019dc78edc88653d771667c0c561f5f
```

然后从本复现仓库根目录运行：

```bash
mamba activate gnn222
export TU_DATA_ROOT=/server/data/TUD
CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-both 2>&1 | tee cosparsify_tu.log
```

原loader缺数据时按原流程下载，无额外`--download`。固定两数据集×9编码×4宽度×10折，**共720次原生训练**，每数据集360次。当前物理batch固定为配置中的8，不自动继续减batch、改精度、换模型或丢图来规避其他失败。

也可先运行完整矩阵中的少量正式条件，后续继续原目录而不重跑已完成项：

```bash
PROTEIN_DATASET=ENZYMES CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-run --encodings rrwp --poly-dims 8 --folds 0
# 同数据集继续填齐剩余条件：
PROTEIN_DATASET=ENZYMES CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-run
PROTEIN_DATASET=ENZYMES bash tools/run_rebuttal.sh protein-report
```

需要后台跑完整矩阵时，使用按“编码×宽度”隔离的分片调度器。它会先用同一原生环境预热两份TU数据，再把每个数据集拆成36个shard，固定上界内分配到4张GPU，全部shard完成后校验并合并10折报告：

```bash
python tools/run_protein_grid.py \
  --python "$(command -v python)" \
  --reference-root Co-Sparsify \
  --data-root "$TU_DATA_ROOT" \
  --output-root outputs/cosparsify_tu_grid \
  --gpus 0 1 2 3 \
  --workers-per-gpu 1
```

`--workers-per-gpu` 默认1；只有实测原模型显存和主机内存允许时再增加。省略 `--resume` 会拒绝覆盖已有输出，续跑时加上该参数。可先运行 `--dry-run` 检查得到72个shard／720次原生训练的计划。后台启动示例：

```bash
nohup setsid python tools/run_protein_grid.py \
  --reference-root Co-Sparsify --data-root "$TU_DATA_ROOT" \
  --output-root outputs/cosparsify_tu_grid \
  --gpus 0 1 2 3 > cosparsify_tu_grid.log 2>&1 < /dev/null &
```

该运行器的进度写入输出根目录的`run_status.json`；每个shard仍调用原有`protein run`，不复制或改写Co-Sparsify训练流程。最终`protein report --shards`会拒绝不同identity、重复记录、缺失组合或checksum不匹配的产物。

`--reference-root`可指定另一处相同、未修改的官方checkout。每个新TU目录为`$REBUTTAL_OUT/cosparsify_tu/数据集名/`；含`records/`、`predictions/`、`native/`原checkpoint／本地日志，以及最终`protein_report.json`。日志保留原预处理耗时输出；JSON保留原avg_train_time_epoch等指标。旧27／64槽自定义模型结果单独保留，不混入此次比较。

完成项先核对实验身份及产物后跳过；完整汇总再核对各编码记录中的实际数据、索引、初始化、模型参数量和原设置一致。改变请求的编码／宽度／折子集不改变实验身份，允许分批继续同一矩阵。删去旧自定义loader／model／trainer后不生成归档；BREC／机制代码不改，不要求重跑。

## 代码阅读顺序

- BREC：`brec.py` 组织运行，`scheduling.py` 并行调度，`encodings.py` 生成通道，`readouts.py` 作判别。
- TU：`protein.py`调用原始入口并保存来源及结果，`protein_encodings.py`只替换编码，`protein_report.py`汇总原始评分；模型、loader和训练器均来自外部Co-Sparsify checkout。

每次运行的 `manifest.json` 记录配置、输入、代码和环境身份。仓库不再保留旧结果表、静态源码清单或归档导出工具。

## BREC 代码验证

```bash
python -m unittest discover -s tests -p 'test_brec_*.py' -v
```

测试覆盖区间运算的精确包络、独立分数公式、两种基与全部默认 Q 预算、累积均值矩、三元提升、重复行匹配、非自逆重标号、官方 reliability 映射、并行与重复运行一致性、内存限制、中断清理、原子保存及续跑身份。配置错误在启动工作进程前拒绝；复现检查同时核对条件覆盖、记录身份和汇总一致性。这些测试使用小图，不替代正式 BREC 全部条件的运行与结果审查。
