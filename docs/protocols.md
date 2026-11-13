# 计算协议与实现

| 实验 | 主要代码 | 数值和读出 |
|---|---|---|
| 精确机制对照 | `rebuttal_experiments/mechanisms.py` | Q=1 的完整有序三元组；精确分数；多重集和均值矩 |
| 补充 BREC | `rebuttal_experiments/brec.py` | 完整二元域；正则图用分数，其他图用区间；多重集或均值矩 |
| TU预测 | `protein.py`、`protein_encodings.py`及外部Co-Sparsify | 原SPPGN1全流程，只替换float32的pair_x／loop_x编码 |
| PPGN | `brec/ppgn_models.py`、`ppgn_realization.py` | 指定的对称幂输入、pair/triple backbone 和神经读出 |
| 更多对象区分任务 | `discriminability/` | 指定宽度、轨道／FWL 参考、共同方向比较和 MIS |

## 补充实验的共同范围

- 完整多重集保留各行出现次数。区间重叠不能证明精确相等；兼容匹配不存在才能确认多重集不同。
- BREC 的未确定、资源受限和未执行项留在分母中；检出下界较大不自动证明真实表达力更强。
- 对象域使用跨通道一致的方向交换，不逐通道独立挑方向。MIS 最优只说明数值相似图求解完成，不认证结构特征的真实值。

- TU完全沿用Co-Sparsify commit 505f6b50f019dc78edc88653d771667c0c561f5f；PROTEINS_full与ENZYMES、9编码、宽度8/12/14/16、10折，共720次原生训练。原SPPGN1、节点属性默认、真实batch、AdamW／warmup／cosine、val/acc选模和评估器不改。
- 同一留出折用于checkpoint选择和最终评分；按用户明确指示不另设独立test，均值及总体标准差按原10折口径报告。只改编码变换并额外保存本地记录，不用此前自定义GNN、64槽或独立test流程。
- joint用三元共享pivot的固定点对读出，通过原pair_x／loop_x接入，不加可训练模块。joint保留endpoint；powers是已有通道的逐元素平方；joint_mix是完整可逆通道换基。有限读出不等同于完整三元字典或完整3-FWL。

## PPGN 输入和模型

`FixedInput.array` 构造 I,S,…,S⁸，其中 S=D⁻¹ᐟ²AD⁻¹ᐟ²。三元输入拼接 (u,v)、(u,w)、(v,w) 三个平面；输入本身没有共享 pivot 操作。它不同于包含六个有序坐标提升的固定编码字典。

模型每层对各分支作仿射映射；pair 分支矩阵相乘，triple 分支收缩共同 pivot；随后作指定有符号根变换、拼接、仿射投影和 BatchNorm。图读出使用代码中定义的对角及边际统计。准确设置以 `configs/ppgn_protocol.json` 为准。

## 确定性与重标号

`brec/determinism.py` 只作用于 BREC，固定数值库线程、随机种子和 PyTorch 严格确定性策略。相同顺序输入的重复运行一致，不等于换编号后仍一致；浮点结果也不能直接作为精确表达力证据。补充 BREC 单独保留同图重标号及官方 reliability 对照，其范围见[运行说明](rebuttal_experiments.md)。
