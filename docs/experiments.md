# 其他实验入口

Rebuttal 补充实验优先使用[统一运行说明](rebuttal_experiments.md)。以下命令用于保留的更广数据集和模型流程；运行前按 [installation.md](installation.md) 准备对应环境。仓库没有附带历史结果，所有报告来自指定的新输出目录。

## 对象区分

数据集与对象域配置在 `configs/object_datasets.json`：

```bash
python reproduce.py list
python reproduce.py prepare --dataset qm9 --data-root data/QM9 --topology topology/qm9.npz
python reproduce.py objects --dataset qm9 --topology topology/qm9.npz \
  --out outputs/objects/qm9 --workers 12 --memory-gib 100
python tools/summarize_object_results.py outputs/objects/qm9 \
  --tag rerun --out outputs/qm9_summary.json
```

按服务器资源调整 `--workers` 和 `--memory-gib`。同代码、输入和配置下可用 `--resume` 继续；修改科学设置后换输出目录。数据集加载不能替代对输入覆盖范围的核对。

## BREC 编码与 PPGN

带精确／区间判别的二元 BREC 使用 `bash tools/run_rebuttal.sh brec`。保留的浮点编码与 PPGN 入口为：

```bash
python reproduce.py brec-plan --brec-file data/brec_v3.npy --trust-pickle --root outputs/brec_model_inputs
python reproduce.py brec-run --root outputs/brec_model_inputs --workers 12 --memory-gib 100
python reproduce.py brec-report --root outputs/brec_model_inputs

python reproduce.py ppgn --brec-file data/brec_v3.npy --trust-pickle \
  --out outputs/ppgn --gpus 0 1 --physical-batch 32
```

配置分别在 `configs/brec_protocol.json` 和 `configs/ppgn_protocol.json`。这些 BREC 入口继续强制确定性算法、单线程数值库和固定随机种子；PPGN 固定物理 batch、dtype、GPU 分配及任务次序，不按运行速度或 OOM 改变计算路径。资源不够时明确失败，调整设置后重新运行。

浮点编码／神经输出和精确区间判别是不同实验，不能互相替代。具体通道、对象域和读出见[协议说明](protocols.md)。
