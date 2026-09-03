"""统一命令行入口。

示例：
    python run.py --dataset industrial --model asenet
    python run.py --dataset taobao --model din
"""
import argparse

from asenet.config import DATASETS, MODELS
from asenet.runner import run_model


def main():
    parser = argparse.ArgumentParser(description="ASENet CTR 预估实验")
    parser.add_argument("--dataset", choices=sorted(DATASETS.keys()), required=True,
                        help="数据集：industrial / taobao")
    parser.add_argument("--model", choices=sorted(MODELS.keys()), required=True,
                        help="模型名，如 dnn / din / dien / asenet 等")
    args = parser.parse_args()
    run_model(args.dataset, args.model)


if __name__ == "__main__":
    main()
