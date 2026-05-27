# CV-Course-Age-Estimation

XJTU CV 大实验 — Experiment 1：基于 **DenseNet121** 和 **ViT-B/16** 的人脸年龄估计（数据集：APPA-REAL）。

## Quick start

```bash
pip install -r requirements.txt
# 把 APPA-REAL 解压到 ./data/appa-real-release/，或用 --data_root 指定路径
python main.py --model densenet     # 训练 + 测试 DenseNet
python main.py --model vit          # 训练 + 测试 ViT
```

## 项目结构

```
repo/
├── src/             核心模块（dataset / models / train / eval / utils）
├── configs/         模型超参数 YAML
├── scripts/         shell 入口
├── notebooks/       Jupyter notebook（探索、可视化）
├── reports/         实验报告 + 图
├── results/         训练日志和指标 CSV
├── checkpoints/     模型权重（不入 git）
├── data/            数据集 placeholder（不入 git，从服务器 /root/autodl-tmp/data 读）
├── main.py          一键运行入口
├── requirements.txt
└── README.md
```

## Dataset

APPA-REAL — http://chalearnlap.cvc.uab.es/dataset/26/data/45/description/

3 个 split（train/valid/test），每个 split 自带 `gt_avg_*.csv` 标签（apparent_age_avg）。

## 模型

| 模型 | 参考 | head 替换 | 优化器 / lr | epochs |
|---|---|---|---|---|
| DenseNet121 | Huang et al., CVPR 2017 | `Linear(1024 → 1)` | Adam, 1e-4 | 25 |
| ViT-B/16 | Dosovitskiy et al., ICLR 2021 | `Linear(768 → 1)` | AdamW, 5e-5 + cosine | 25 |

评价指标：**MAE**（平均绝对误差，单位"岁"）+ **RMSE**。

## Authors

XJTU CV 大实验 (2026, 苏远歧)。Mike — Enhui Li (GitHub: Lichangheng). 队友：TBA。
