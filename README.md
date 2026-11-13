# Graph Structural Encodings: Operations, Budgets, and Readouts

实验代码与 rebuttal 补充实验。BREC／机制以Python 3.12、Linux为目标环境；TU使用Co-Sparsify的loader／Lightning训练与评估流程，并将骨干模型替换为`gnnplus_official`的GNNPlus GINE。所有命令从本目录运行；数据路径和输出目录由运行者指定。

## 运行补充实验

环境、输入和完整命令见[补充实验说明](docs/rebuttal_experiments.md)。每个命令只执行指定任务：

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec

CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-both
```

运行前按说明设置 `BREC_FILE`、`TU_DATA_ROOT` 等路径。各实验进度见补充实验说明。
`protein-both`使用Co-Sparsify的TU数据、十折任务和训练器，并将模型替换为GNNPlus官方GINE层、SAN图头和mean pooling；编码通过节点loop通道与有向边pair通道注入。PROTEINS_full与ENZYMES各9编码×4宽度×10折，总计720次训练；同一留出折选checkpoint并评分。结果保存在outputs/rebuttal/cosparsify_tu。

完整矩阵可用`python tools/run_protein_grid.py`在后台分片运行；先加`--dry-run`检查任务计划。运行器支持4 GPU调度、内存余量、断点续跑、跨shard身份与checksum校验，以及最终合并报告。

## 阅读代码

| 文件 | 职责 |
|---|---|
| `rebuttal_experiments/__main__.py` | 选择实验入口 |
| `mechanisms.py` | 两项精确机制对照 |
| `brec.py` | BREC 配置循环、重标号对照及结果汇总 |
| `scheduling.py` | BREC 多进程调度、CPU 绑定及总内存预算监控 |
| `encodings.py`、`intervals.py`、`readouts.py` | 编码公式、区间算术和判别规则 |
| `protein.py` | 原Co-Sparsify入口调用、完成项复用和只读结果捕获 |
| `protein_encodings.py` | 只替换原poly变换；相同宽度的9种编码 |
| `protein_gnnplus.py` | GNNPlus官方GINE／SAN头的输入适配与模型构造 |
| `protein_report.py` | 原始10折分数与同宽度配对差汇总 |
| `common.py` | 文件读写、运行配置和资源限制 |

表内短文件名均位于 `rebuttal_experiments/`。建议先读实验入口，再读对应计算函数；TU不重写训练或最终评分流程。

其他可复用代码保留在 `brec/`、`discriminability/`、`downstream/` 和 `gnnplus_official/`。更广的数据集及 PPGN 命令见[其他实验入口](docs/experiments.md)，计算范围见[协议说明](docs/protocols.md)。第三方归属见 [THIRD_PARTY.md](THIRD_PARTY.md)。
