"""ASENet CTR 预估实验复现代码库。

将原先 26 个几乎完全复制的独立脚本重构为共享包结构，模型逻辑与原始代码逐行一致。
运行方式：
    python run.py --dataset industrial --model din      # 统一 CLI
    python Industrial_DIN.py                             # 薄封装（等价）
"""
