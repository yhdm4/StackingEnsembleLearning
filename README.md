# StackingEnsembleLearning

Regression stacking experiments for Sections 4.1–4.4: six datasets, five outer folds, twelve comparison methods.

**当前状态（2026-10-12）：PU纯Sigmoid网络已按原五折划分从1000轮改为2000轮，两种输入各重新训练五折。4.1/4.2/4.3/4.4仍为90/180/360/360条、合计990条；4,950项指标独立反算通过，最大误差5.68e-14。十二方法比较保持原值，Proposed训练第3、测试第1。**

## 本次PU更新与论文编号

代码和结果文件沿用实验编号41、42、43、44，分别对应当前论文4.2、4.3、4.4、4.5；当前论文4.1为实验设置。PU的2000轮仅适用于纯Sigmoid网络，十二方法比较中的PU仍为1200轮。

本次重新训练十个网络，更新15条PU的MLP折级记录（41中5条、42中10条），以及PU曲线、均值和样本标准差、相关统计与论文百分比。其余折级记录逐行保留。完整更新证据见results/pu_2000epoch_update.json；模型和逐样本预测保存在本地独立运行目录，不上传大型训练缓存。

只重训PU两种输入可运行：python main.py --section 42 --dataset parkinsons_updrs_regression --output results/reproduced_pu2000_only。只选一个数据集时结果保存在该输出目录的五折缓存，仓库已发布的完整990条结果不会被这一部分运行覆盖。参数或代码指纹变化时应使用新目录，避免旧1000轮完成标记被当作新结果。

## 文件结构

根目录仅保留五个Python主程序。所有新功能直接加入这些原文件，各函数以中文说明用途。

| 文件 | 用途 |
|---|---|
| main.py | 数据去重、训练折清洗、基模型OOF、多层感知机、凸权重求解、内部轮数选择、配对诊断、五折调度与完整性门槛 |
| train_4_1.py | 纯Sigmoid MLP、LR与Ridge训练拟合比较及五折曲线 |
| ablation_4_2.py | 仅基模型预测与加入原始特征的消融；复用4.1联合输入结果 |
| train_4_3.py | 十二方法训练、保存、重载核对与训练指标 |
| predict_4_4.py | 重载4.3模型，评估外层测试折；前期选参曾查看第0折测试结果 |
| data/ | 六个输入CSV；AD已脱敏，SC为官方完整原始数据 |
| results/ | 正式折级指标、汇总、曲线与统计；默认在这里覆盖输出 |

## 环境与运行

原实验环境保持一致：Python 3.11.9、NumPy 1.26.4、pandas 2.0.3、scikit-learn 1.5.0、XGBoost 2.0.3、PyTorch 2.0.1+cu118、joblib 1.4.2；绘图使用Matplotlib 3.7.2，统计使用SciPy 1.16.2。

```powershell
# 当前机器：缺失依赖在工作区安装，不修改系统Python配置。
$env:PYTHONPATH='C:\Users\86133\Desktop\论文英文翻译\tmp\experiment_dependencies'
# 使用当前冻结配置创建新的复现实验；再次执行相同命令可续跑该目录。
& 'C:\ProgramData\anaconda3\python.exe' -B -X utf8 main.py --output results/reproduced_pu2000
```

其他电脑装好依赖后直接`python main.py`。默认输出覆盖`results/`，模型在`results/.cache/fold0`至`fold4`；可用`--output`指定独立输出、`--cache-prefix`指定已有折目录前缀。`--section 41/42/43/44`、`--dataset`和`--fold`可选择部分计算。4.4须在相同目录完成4.3后运行。协议文件记录代码、数据、参数和环境指纹；出现不一致即拒绝复用，不静默混用新旧缓存。

```bash
python main.py --summarize --input results/summary
python main.py --plot --input results/curves
```

`--summarize`先检查四节完整性，再生成正式排名和统计。未齐全时只有折目录里的过程结果，没有正式综合排名。

## 数据及去重规则

| 缩写 | CSV文件名 | 原始行数 | 合并后行数 | 特征数 |
|---|---|---:|---:|---:|
| AD | AlignedData_aligned | 9470 | 9364 | 18 |
| CO | Concrete_Data | 1030 | 996 | 8 |
| PU | parkinsons_updrs_regression | 5875 | 5875 | 18 |
| WQ | wine_quality_combined | 6497 | 5320 | 12 |
| AS | airfoil_self_noise | 1503 | 1503 | 5 |
| SC | superconductivity | 21263 | 15170 | 81 |

去重以**实际输入完全相同**为准，目标取组内原始记录的算术均值；不按组大小加权。保留组首次出现顺序，划分前去重。每折保存`group_mapping.csv`（原始行对应输入组）和`data_audit.json`（数据哈希、原始数、组数、冲突目标组数）。这会改变估计对象为“唯一输入位置上的平均响应”，不只是删掉逐列完全相同的整行。

AD的内部文件名表示对齐后的工业过程记录，不是材料名称；只公开x1–x18与y，省略时间戳和无关字段，脱敏不改变实际建模数值。WQ包括区分红白葡萄酒的输入列。SC来自[UCI官方数据](https://archive.ics.uci.edu/dataset/464/superconductivty+data)，保留number_of_elements；替换来源不明的16244行旧副本。

PU和AS无重复输入，既有十二方法比较保留。2026-10-12仅对PU纯Sigmoid网络的两种输入进行2000轮五折重训；复用前已逐折核验数据、样本索引、标准化输入/目标以及30个基模型的折外预测，最大OOF差异为0。其他数据集保留已发布结果。

## 实验协议

1. 固定KFold(5,shuffle=True,random_state=42)。每次80%训练、20%测试，测试折互斥。
2. 只在训练集做目标IQR(α=0.8)和随后3σ筛选，再拟合Z-score标准化器（均值0、标准差1）。测试集不清洗。
3. 清洗后做十折基模型OOF（out-of-fold，折外预测：每个样本的预测来自未用该样本训练的模型）。外层训练预处理器在十折之前拟合，这不是内层逐折重新拟合预处理器的协议。
4. 第一层SVM用C=100、gamma=auto、epsilon=0.1；RandomForest用200棵树；XGBoost用250棵树、学习率0.05、最大深度6，随机种子42，CPU单线程。
5. 测试输入取每类十个保留模型预测的均值。三列预测与原始特征拼接后输入Proposed。Ridge的L2正则化系数为1、截距不惩罚。
6. 4.1/4.2用纯Sigmoid网络，4.3/4.4用BatchNorm（批归一化）、LeakyReLU、Dropout（训练中随机屏蔽神经元）与AdamW。参数集中在main.py中的FIXED_MLP_CONFIGS和FIT_CONFIGS。

十二方法：Proposed、NN-Stacking、Convex-Stacking、Meta-XGB、Meta-RF、Meta-SVM、Meta-LR、Meta-Ridge、E2E-XGB、E2E-RF、E2E-SVM、E2E-MLP。

Meta方法接收原始特征加三列基模型预测，E2E方法只接原始特征。NN-Stacking由原始特征产生三个无约束、随样本变化的权重及修正项，计算sum(theta_j(x)*p_j)+phi(x)。Convex-Stacking只用OOF预测拟合三个全局固定权重，权重非负且和为1，无截距、无修正项、无原始特征；使用原始目标尺度的最小二乘目标，通过枚举单纯形全部非空面求解。它是文献6非负堆叠和文献7Super Learner凸组合思想的统一对照，并非两篇文献原算法的逐项复现。

## 统一神经参数与历史诊断

4.3/4.4按数据集冻结以下完整配置；五折以及Proposed、E2E-MLP、NN-Stacking共用。训练预算固定，预算内按训练评估MSE保存最优检查点，最优检查点轮数可以不同。4.1/4.2纯Sigmoid配置由FIT_CONFIGS与FIT_HIDDEN_DIMS独立冻结；PU为2000轮，隐藏层1280,1024,512,256、学习率0.0004、批大小256，其余设置保留。

| 数据集 | 隐藏层 | Dropout | Weight decay | 学习率 | Batch size | Epochs |
|---|---|---:|---:|---:|---:|---:|
| AD | 1536,1024,512,256 | 0.02 | 0.000001 | 0.0003 | 512 | 2500 |
| CO | 384,192,96 | 0.05 | 0.00001 | 0.0008 | 32 | 750 |
| PU | 1280,1024,512,256 | 0.40 | 0.0002 | 0.0004 | 256 | 1200 |
| WQ | 768,512,256,128 | 0.35 | 0.0003 | 0.0003 | 64 | 150 |
| AS | 512,256,128 | 0.03 | 0.000001 | 0.0003 | 32 | 1800 |
| SC | 2048,1536,768 | 0.10 | 0.0001 | 0.0003 | 256 | 280 |

第0折测试结果曾参与前期参数比较，因此当前五折是选参后的描述性复核，不是独立泛化估计。训练循环不使用测试标签；严谨的独立性能验证仍需新增留出数据或完整嵌套交叉验证。EPOCH_CANDIDATES为空，本轮不执行逐折内部选轮数。综合排名使用未四舍五入的指标，Proposed训练平均名次3.4667（第3），测试平均名次3.9750（第1）。

上一轮额外诊断以CO固定500轮、SC固定140轮比较保留重复训练记录和合并均值训练记录。相同的唯一输入组划分、同一组均值测试标签；重复分支的OOF也按输入组分折。SC两分支均使用官方81特征。历史诊断保存在各数据集`dedup_diagnostic/`，历史内部选轮数保存在`epoch_selection/`；不并入本轮十二方法排名，也不代表本轮统一参数选择过程。

## 结果计数与指标

| 节 | 记录数 | 含义 |
|---|---:|---|
| 4.1 | 6×5×3=90 | 三种元学习器 |
| 4.2 | 6×5×3×2=180 | 三种元学习器×两种输入 |
| 4.3 | 6×5×12=360 | 十二方法训练拟合 |
| 4.4 | 6×5×12=360 | 同一批模型测试预测 |

总计990条，每条包含一个数据集、一个折、一个方法/输入的整组指标，不按样本或指标个数重复计数。

R²越高越好；MAPE（平均绝对百分比误差）、MAE（平均绝对误差）、RMSE（均方根误差）越低越好，均以原目标尺度计算。MSELoss为标准化目标上的均方误差。MAPE对接近零的目标采用1e-8分母保护。

排名使用未四舍五入的数值，每种方法平均5折×6数据集×4指标＝120个名次；这120项不是统计独立的数据集。Friedman检验以六数据集的五折均值为区组，检验方法整体差异；Kendall W描述同一数据集五折排名的一致性。置换P值使用固定种子，Holm法校正多重检验；重叠训练折对应的Kendall条件P值只作探索参照。

`fit_seconds`是各训练步骤的记录时间；`prediction_seconds`目前包含模型重载与预测，不应称为纯推理延迟；共享基模型预测另列，不能给每种方法重复相加。PU/AS沿用的历史记录没有完整分项耗时，对缺测项目应标缺失，不能当作0秒。

图1保留完整训练横轴、普通线性纵轴，无平滑，主MLP曲线约占画面2/3；允许起始少量尖峰超出纵轴范围，保留LR/Ridge放大小图。

代码和数据仓库：[yhdm4/StackingEnsembleLearning](https://github.com/yhdm4/StackingEnsembleLearning)。本仓库由 Ziqi Wang（王子骐）维护，保存论文实验代码、数据与结果。


