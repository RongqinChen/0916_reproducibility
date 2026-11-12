# Graph Structural Encodings: Operations, Budgets, and Readouts

实验代码与 rebuttal 补充实验。Python 3.12、Linux 为目标环境，所有命令从本目录运行。仓库不包含旧实验结果；数据路径和输出目录由运行者指定。

## 运行补充实验

环境、输入和完整命令见[补充实验说明](docs/rebuttal_experiments.md)。每个命令只执行指定任务：

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec

PROTEIN_DEVICE=cuda:0 bash tools/run_rebuttal.sh protein-both
```

运行前按说明设置 `BREC_FILE`、`TU_DATA_ROOT` 等路径。各实验进度见补充实验说明。
`protein-both` 依次运行 PROTEINS 和 ENZYMES，各比较 9 种编码、135 个配对训练任务，共 270 个。覆盖零结构编码、polynomial、endpoint、Q=1/2 joint 和完整换基；统一 64 个结构槽及共同读出，结果写入 `outputs/rebuttal/protein_encodings/数据集名/`。需要下载 TU 原始数据时加 `--download`。全部设置共享一次特征准备，保留全部有序三元组；报告同时给出预设比较与仅按验证集选择配置的测试结果。

## 阅读代码

| 文件 | 职责 |
|---|---|
| `rebuttal_experiments/__main__.py` | 选择实验入口 |
| `mechanisms.py` | 两项精确机制对照 |
| `brec.py` | BREC 配置循环、重标号对照及结果汇总 |
| `scheduling.py` | BREC 多进程调度、CPU 绑定及总内存预算监控 |
| `encodings.py`、`intervals.py`、`readouts.py` | 编码公式、区间算术和判别规则 |
| `protein.py` | 蛋白质实验的命令参数 |
| `protein_encodings.py` | 编码清单、完整换基、通道保留关系与预设比较 |
| `protein_data.py`、`tu_data.py` | 三元特征、数据准备、训练输入及 TU 数据读取 |
| `protein_model.py`、`protein_training.py` | 共同模型、配对训练及验证集选模 |
| `protein_results.py` | 测试预测与配对结果汇总 |
| `common.py` | 文件读写、运行配置和资源限制 |

表内短文件名均位于 `rebuttal_experiments/`。建议先读实验入口，再读对应计算函数；训练与测试预测分别执行。

其他可复用代码保留在 `brec/`、`discriminability/`、`downstream/` 和 `gnnplus_official/`。更广的数据集及 PPGN 命令见[其他实验入口](docs/experiments.md)，计算范围见[协议说明](docs/protocols.md)。第三方归属见 [THIRD_PARTY.md](THIRD_PARTY.md)。
