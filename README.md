# Graph Structural Encodings: Operations, Budgets, and Readouts

实验代码与 rebuttal 补充实验。BREC／机制以Python 3.12、Linux为目标环境；TU使用Co-Sparsify原环境，所有命令从本目录运行。仓库不包含旧实验结果；数据路径和输出目录由运行者指定。

## 运行补充实验

环境、输入和完整命令见[补充实验说明](docs/rebuttal_experiments.md)。每个命令只执行指定任务：

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec

CUDA_VISIBLE_DEVICES=0 bash tools/run_rebuttal.sh protein-both
```

运行前按说明设置 `BREC_FILE`、`TU_DATA_ROOT` 等路径。各实验进度见补充实验说明。
`protein-both`直接运行未修改的Co-Sparsify TU入口，只有编码不同。PROTEINS_full与ENZYMES，各9编码×4宽度(8/12/14/16)×10折，总计720次原生训练；同一留出折选checkpoint并评分，报告原始均值与总体标准差。使用原SPPGN1、loader和训练器，结果保存在outputs/rebuttal/cosparsify_tu。环境、固定commit及分批运行见说明。

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
| `protein_report.py` | 原始10折分数与同宽度配对差汇总 |
| `common.py` | 文件读写、运行配置和资源限制 |

表内短文件名均位于 `rebuttal_experiments/`。建议先读实验入口，再读对应计算函数；TU不重写训练或最终评分流程。

其他可复用代码保留在 `brec/`、`discriminability/`、`downstream/` 和 `gnnplus_official/`。更广的数据集及 PPGN 命令见[其他实验入口](docs/experiments.md)，计算范围见[协议说明](docs/protocols.md)。第三方归属见 [THIRD_PARTY.md](THIRD_PARTY.md)。
