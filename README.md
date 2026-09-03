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

- 依赖见 `requirements.txt`（TensorFlow 2.x / scikit-learn / numpy）。
- 数据与模型输出路径硬编码在 `asenet/config.py` 的 `DATASETS` 中，可按需修改。
- 本机未安装 TensorFlow 与实验数据，无法真实跑训练；请在有数据/环境机器上运行以复现论文指标。
