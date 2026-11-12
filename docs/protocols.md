# 计算协议与实现

| 实验 | 主要代码 | 数值和读出 |
|---|---|---|
| 精确机制对照 | `rebuttal_experiments/mechanisms.py` | Q=1 的完整有序三元组；精确分数；多重集和均值矩 |
| 补充 BREC | `rebuttal_experiments/brec.py` | 完整二元域；正则图用分数，其他图用区间；多重集或均值矩 |
| 蛋白质预测 | `rebuttal_experiments/protein_data.py`、`protein_model.py` | float64 特征缓存、float32 模型输入；完整行交互后聚合 |
| PPGN | `brec/ppgn_models.py`、`ppgn_realization.py` | 指定的对称幂输入、pair/triple backbone 和神经读出 |
| 更多对象区分任务 | `discriminability/` | 指定宽度、轨道／FWL 参考、共同方向比较和 MIS |

## 补充实验的共同范围

- 完整多重集保留各行出现次数。区间重叠不能证明精确相等；兼容匹配不存在才能确认多重集不同。
- BREC 的未确定、资源受限和未执行项留在分母中；检出下界较大不自动证明真实表达力更强。
- 对象域使用跨通道一致的方向交换，不逐通道独立挑方向。MIS 最优只说明数值相似图求解完成，不认证结构特征的真实值。
- 蛋白质九组共同使用 64 个结构槽，完整保留选定编码并补零，不作截断或随机投影；共享backbone、三元组节点表示、MLP读出和更新预算。`none`只将结构槽置零，仍保留共同读出。旧27槽实验不与新版混为配对条件。
- 蛋白质实验固定 PROTEINS 与 ENZYMES，各 5 folds × 3 seeds × 9 variants，共270个任务。覆盖零结构编码、polynomial Q=1/2、nonjoint增广、endpoint Q=2、joint Q=1/2，以及(I,I+M)和Chebyshev完整换基；不把有限字典称为完整3-FWL。
- 主比较为joint_q2对endpoint_q2：保留后者全部通道，再增加27个joint通道。完整joint Q=2也保留Q=1 joint；这些是信息保留关系，不是每个任务上必然涨分的结论。完整换基检验相同信息的利用，不算新增信息。
- 全有序三元组含重复顶点；float64特征分块写盘，float32训练输入按块读取和重计算，不采样或删去大图。缓存25个polynomial和27个joint通道；endpoint按需生成，完整换基按稀疏线性变换派生。训练集标准化由逐图通道和／Gram矩阵合并，不扫描全部缓存。
- 蛋白质标准化只拟合训练记录，checkpoint只按验证NLL选择，测试预测单独执行。全部预设配置都报告；另按每fold／seed的验证NLL选编码，独立汇总其外层测试结果，不按测试成绩挑选配置。fold×seed不作为独立数据集重复。

## PPGN 输入和模型

`FixedInput.array` 构造 I,S,…,S⁸，其中 S=D⁻¹ᐟ²AD⁻¹ᐟ²。三元输入拼接 (u,v)、(u,w)、(v,w) 三个平面；输入本身没有共享 pivot 操作。它不同于包含六个有序坐标提升的固定编码字典。

模型每层对各分支作仿射映射；pair 分支矩阵相乘，triple 分支收缩共同 pivot；随后作指定有符号根变换、拼接、仿射投影和 BatchNorm。图读出使用代码中定义的对角及边际统计。准确设置以 `configs/ppgn_protocol.json` 为准。

## 确定性与重标号

`brec/determinism.py` 只作用于 BREC，固定数值库线程、随机种子和 PyTorch 严格确定性策略。相同顺序输入的重复运行一致，不等于换编号后仍一致；浮点结果也不能直接作为精确表达力证据。补充 BREC 单独保留同图重标号及官方 reliability 对照，其范围见[运行说明](rebuttal_experiments.md)。
