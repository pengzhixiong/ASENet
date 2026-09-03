# ASENet

ASENet（Adversarial Sequential Expert Network）的 CTR 预估实验复现代码，对应论文
`ASENet_paper_latex/ASENet.tex`。原始 26 个独立脚本已重构为共享包结构，**模型数值逻辑与
原始代码逐行一致**（原脚本归档于 `legacy/`）。

## 目录结构

```
ASENet/
├── run.py                    # 统一命令行入口
├── Industrial_*.py           # 28 个薄封装脚本（保留原运行方式）
├── Taobao_*.py
├── asenet/
│   ├── config.py             # 特征表 / 数据集 / 模型元信息（含 Taobao 两套词表变体）
│   ├── data.py               # CSV 数据加载与标签读取
│   ├── evaluate.py           # AUC / Logloss / PCOC 计算
│   ├── serve.py              # serving 签名与模型保存
│   ├── base.py               # 模型基类（Hashing/Embedding 初始化）
│   ├── layers.py             # 复用自定义层
│   ├── runner.py             # 训练/评估/导出编排
│   └── models/               # 13 个模型实现
└── legacy/                   # 原始脚本（对照参考）
```

## 运行方式

```bash
# 统一 CLI
python run.py --dataset industrial --model asenet
python run.py --dataset taobao --model din

# 等价：薄封装脚本（保留原文件名）
python Industrial_DIN.py
python Taobao_ASENet.py
```

## 支持的模型

| 模型 | key | 类型 |
|------|-----|------|
| DNN | `dnn` | 静态 |
| DeepFM | `deepfm` | 静态 |
| DIN | `din` | 序列 |
| DIEN | `dien` | 序列 |
| DMR | `dmr` | 序列 |
| FINAL | `final` | 长序列 |
| TWIN | `twin` | 长序列 |
| MIRRN | `mirrn` | 长序列 |
| GAN(Naive) | `gan` | 生成式 |
| QNN-α | `qnn` | 生成式/SOTA |
| DLF | `dlf` | 生成式/SOTA |
| HierDiffuse | `hierdiffuse` | 生成式/SOTA |
| **ASENet** | `asenet` | 本文方法 |

## ASENet 模型架构

ASENet（Adversarial Sequential Expert Network）把 CTR 预估形式化为一个**条件分布对齐**问题，用
对抗训练范式来训练。它由三个核心模块构成，分别对应论文提出的三个挑战（CH1–CH3）：

| 挑战 | 问题 | ASENet 的对策 | 对应模块 |
|------|------|--------------|---------|
| CH1 | 用户兴趣的时序动态演化 | LSTM 时序编码 + 共享嵌入 | 时序意图生成器 TIG |
| CH2 | 判别器在异构长尾分布下的瓶颈 | MoE 软路由 + 负载均衡 | 自适应路由判别器 ARD |
| CH3 | 离散特征空间下对抗训练不稳定 | Wasserstein 距离 + 梯度惩罚 | 分布对齐目标 WDA |

整体数据流如下：

```
                        ┌──────────── 特征嵌入与共享模块 ────────────┐
输入 x ─► Hashing ─► Embedding ─► 静态特征 e_stat ─┐
             │  (dim=8)              目标/序列同类 ID 共享同一 Embedding │
             │                                                │
             └► 序列特征 split("^") ─► LSTM ─► h_T ───────────┴─► [e_stat; h_T]
                                                                      │
                              ┌───────────────────────────────────────┘
                              ▼
                  ┌── 时序意图生成器 TIG ──┐
                  │   MLP [128,64,1]→sigmoid│ ──► ŷ = G(x)
                  └─────────────────────────┘
                              │
                              │  (特征 x, 预测 ŷ)  与  (特征 x, 真实 y)
                              ▼
                  ┌── 自适应路由判别器 ARD ──┐
                  │ [e_stat; h_T; ℓ] → MoE(K=4)│ ──► D(x,ℓ)=Σ π_k·E_k
                  └───────────────────────────┘
                              │
                              ▼
                  ┌── 联合对抗优化 WDA ──┐
                  │  Wasserstein + GP + 负载均衡│
                  └──────────────────────┘
```

### 1. 时序意图生成器 TIG（Generator）

TIG 即 CTR 预估模型本身（推理时只使用它），负责把高维稀疏 ID 特征映射为稠密嵌入、建模用户
兴趣的时序演化，最终输出点击概率。

**统一语义空间（共享嵌入）。** 目标商品与用户历史行为序列中的同类 ID 共享同一张 Embedding
表，强制二者落在同一隐空间流形上：

| 数据集 | 共享组 | base 特征（目标侧） | 成员特征（序列侧） |
|--------|--------|--------------------|--------------------|
| Industrial | `cate` | `cate_disp_id` | `search_cate_disp_list_30d` |
| Industrial | `seller` | `seller_id` | `search_seller_list_30d` |
| Taobao | `cate` | `cate_id` | `cate_his` |
| Taobao | `brand` | `brand` | `brand_his` |

**时序兴趣提取。** 序列特征按分隔符 `^` 切分，截断/补齐到 `max_seq_len`（Industrial=20 /
Taobao=50），经 Hashing→Embedding 后送入 LSTM（`units=dim=8`），取最后隐状态 `h_T` 作为时序
意图表示。

**输出。** `h_T` 与静态特征向量 `e_stat` 拼接后，通过三层 MLP（`[128, 64, 1]`，ReLU + sigmoid）
输出 CTR 概率：

```
ŷ = G(x) = σ( MLP( [e_stat ; h_T] ) )
```

### 2. 自适应路由判别器 ARD（Discriminator）

判别器判断 `(特征, CTR)` 对来自真实分布还是生成分布，为生成器提供分布级反馈。为应对工业广告
数据的极端异构性，ARD 用 MoE 取代单一 MLP。

**输入。** `z = [e_stat ; h_T ; ℓ]`，其中 `ℓ` 为真实标签 `y` 或生成预测 `ŷ`（标量）。

**专家网络。** `K=4` 个并行的前馈专家网络，每个专家结构为 `[64, 32] → 1`（ReLU 隐藏层 + 线性
输出），参数完全解耦，各自在特定数据子流形上学习判别函数。

**软门控路由。** 门控网络用 Softmax 对每个样本输出路由权重 `π(z) ∈ R^K`：

```
π(z) = Softmax( W_g·z + b_g )
```

**混合输出。** `D(x, ℓ) = Σ_k π_k(z)·E_k(z)`。反向传播时梯度也按门控权重分流——当前样本主要
由「最懂它」的专家提供梯度，从而缓解梯度冲突。

### 3. 分布对齐目标 WDA（WGAN-GP）

为避免原生 GAN 的 JS 散度在离散高维稀疏空间下退化导致的梯度消失/模式崩塌，ASENet 采用
WGAN-GP。

**判别器损失：**

```
L_D = ( E[D(x,ŷ)] − E[D(x,y)] )          # Wasserstein 距离估计
    + λ · E[ (‖∇_ℓ̂ D‖₂ − 1)² ]            # 梯度惩罚
    + γ · L_balance                        # MoE 负载均衡
```

- 梯度惩罚（`λ = 10.0`）：对真实与生成 CTR 的插值点求判别器梯度范数，逼近期望值 1，软性强制
  1-Lipschitz 连续。
- 负载均衡损失（`γ = 0.2`）：`L_balance = K·Σ_k μ_k² − 1`，其中 `μ_k` 为第 `k` 个专家的批内平均
  门控权重，防止「赢者通吃」式专家坍缩。

**生成器损失（混合目标）：**

```
L_G = α · L_CTR + L_Adv           # α = 20.0
```

- 内容损失 `L_CTR`：标准 BCE，保证局部预测精度（主要决定 AUC）。
- 对抗损失 `L_Adv = −E[ D(x, G(x)) ]`：对齐预测概率流形与真实后验流形（主要改善 Logloss）。

### 关键超参数

| 项 | 值 |
|----|----|
| 优化器 / 学习率 | Adam / 0.001 |
| batch_size / epochs | 2048 / 1 |
| Embedding 维度 `d` | 8 |
| 序列长度 `L` | Industrial 20 / Taobao 50 |
| Generator MLP | [128, 64, 1] |
| 专家数量 `K` / 专家隐层 | 4 / [64, 32] |
| GP 系数 `λ` | 10.0 |
| 负载均衡系数 `γ` | 0.2 |
| BCE 内容损失权重 `α` | 20.0 |

**推理。** 训练收敛后仅导出 TIG（Generator）用于线上服务，ARD 只在训练阶段提供分布反馈，之后
被丢弃。

## 重构保真说明

重构时严格保留了原实验中的非直观差异（这些差异会影响结果，故未做“统一”）：

- **Taobao 两套词表大小**（影响 Hashing 桶数 → Embedding 形状）：大词表用于
  `asenet/dien/dmr/deepfm/final/mirrn/twin`，小词表用于 `din/dlf/dnn/gan/hierdiffuse/qnn`。
- **序列特征启用**：静态模型 `dnn/deepfm/gan` 关闭序列特征。
- **序列/目标特征名映射**与**序列长度**（Industrial=20 / Taobao=50，短期 5 / 10）。
- **`_SHUFFLE_SIZE`**：Industrial 的 `asenet/final/twin/mirrn` 为 10000，其余 100000。
- **`price` 特征处理**：仅 Taobao 存在；不同模型处理方式不同（哈希嵌入 / 原始数值 / 跳过）。
- 模型内部的已知怪癖（如 ASENet 跳过 price、DNN 序列分隔符为 `,`、QNN 的 SE Loss、
  FINAL 的跨 block 蒸馏）均按原样保留。

## 环境说明

- **Python 3.9**，**TensorFlow 2.8**（另需 scikit-learn / numpy，见 `requirements.txt`）。
- 数据与模型输出路径硬编码在 `asenet/config.py` 的 `DATASETS` 中，可按需修改。
- 本机未安装 TensorFlow 与实验数据，无法真实跑训练；请在有数据/环境机器上运行以复现论文指标。
