# Graph Structural Encodings: Operations, Budgets, and Readouts

实验代码与 rebuttal 补充实验。Python 3.12、Linux 为目标环境，所有命令从本目录运行。仓库不包含旧实验结果；数据路径和输出目录由运行者指定。

## 运行补充实验

环境、输入和完整命令见[补充实验说明](docs/rebuttal_experiments.md)。每个命令只执行指定任务：

```bash
bash tools/run_rebuttal.sh mechanisms
bash tools/run_rebuttal.sh brec
bash tools/run_rebuttal.sh objects

bash tools/run_rebuttal.sh protein-prepare
bash tools/run_rebuttal.sh protein-train
bash tools/run_rebuttal.sh protein-predict
bash tools/run_rebuttal.sh protein-report
```

运行前按说明设置 `BREC_FILE`、`MUTAG_TOPOLOGY`、`TU_DATA_ROOT` 等路径。各实验进度见补充实验说明。

## 阅读代码

| 文件 | 职责 |
|---|---|
| `rebuttal_experiments/__main__.py` | 选择实验入口 |
| `mechanisms.py` | 两项精确机制对照 |
| `brec.py` | BREC 配置循环、重标号对照及结果汇总 |
| `scheduling.py` | BREC 多进程调度、CPU 绑定及总内存预算监控 |
| `encodings.py`、`intervals.py`、`readouts.py` | 编码公式、区间算术和判别规则 |
| `objects.py`、`object_comparison.py` | 对象域实验流程、特征构造及比较 |
| `protein.py` | 蛋白质实验的命令参数 |
| `protein_data.py`、`tu_data.py` | 三元特征、数据准备、训练输入及 TU 数据读取 |
| `protein_model.py`、`protein_training.py` | 共同模型、配对训练及验证集选模 |
| `protein_results.py` | 测试预测与配对结果汇总 |
| `common.py` | 文件读写、运行配置和资源限制 |

表内短文件名均位于 `rebuttal_experiments/`。建议先读实验入口，再读对应计算函数；训练与测试预测分别执行。

其他可复用代码保留在 `brec/`、`discriminability/`、`downstream/` 和 `gnnplus_official/`。更广的数据集及 PPGN 命令见[其他实验入口](docs/experiments.md)，计算范围见[协议说明](docs/protocols.md)。第三方归属见 [THIRD_PARTY.md](THIRD_PARTY.md)。
