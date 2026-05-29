# CPU Smoke Test 测试手册

## 1. 测试目的

本次测试用于验证年龄估计实验代码链路在 CPU 环境下是否能够完整跑通。该测试属于 smoke test(小规模冒烟测试)，重点不是追求最终模型性能，而是确认以下流程是否正常：

- 参数解析与配置文件加载
- 数据集读取与 train/val/test 划分
- DenseNet 模型构建
- 预训练权重加载
- 前向传播、反向传播与参数更新
- 验证集评估
- best checkpoint 保存
- checkpoint 加载
- 测试集评估与结果汇总

## 2. 测试命令

```powershell
python main.py --model densenet --mode all --epochs 1 --batch_size 4 --data_root data/appa-real-release --out_dir results/smoke --ckpt_dir checkpoints_smoke
```

## 3. 测试环境与运行配置

本次运行日志中的基础信息如下：

```text
[main] model=densenet | mode=all | device=cpu
[main] config: configs\densenet.yaml
[main] effective cfg: {
  'model': 'densenet121',
  'pretrained': True,
  'epochs': 1,
  'batch_size': 4,
  'lr': 0.0001,
  'weight_decay': 0.0,
  'optimizer': 'adam',
  'loss': 'l1',
  'scheduler': {'name': 'step', 'step_size': 10, 'gamma': 0.5},
  'img_size': 224,
  'num_workers': 4,
  'seed': 42
}
[main] out_dir: results/smoke\densenet | ckpt_dir: checkpoints_smoke
[dataloaders] train=4113 | val=1500 | test=1978 | bs=4 | workers=4
```

关键配置含义：

- `model=densenet`：使用 DenseNet 系列模型。
- `densenet121`：实际骨干网络为 DenseNet-121。
- `mode=all`：执行完整流程，包括训练、验证和测试。
- `device=cpu`：本次运行使用 CPU，没有使用 GPU。
- `pretrained=True`：模型使用预训练权重初始化。
- `epochs=1`：只训练 1 个 epoch，适合作为链路验证。
- `batch_size=4`：每个 batch 读取 4 张图像。
- `lr=0.0001`：学习率为 1e-4。
- `loss=l1`：训练损失为 L1 loss，即平均绝对误差。
- `img_size=224`：输入图像尺寸为 224。
- `num_workers=4`：数据加载使用 4 个 worker。
- `seed=42`：随机种子固定为 42，便于复现。

## 4. 数据集划分

日志显示：

```text
[dataloaders] train=4113 | val=1500 | test=1978 | bs=4 | workers=4
```

含义如下：

- 训练集：4113 张图像
- 验证集：1500 张图像
- 测试集：1978 张图像
- batch size：4

训练进度中显示一轮训练约有 `1028` 个 batch。因为 `4113 / 4 = 1028.25`，所以实际 batch 数与数据量匹配。如果训练 DataLoader 设置了 `drop_last=True`，最后不足一个 batch 的样本可能会被丢弃。

## 5. 训练过程摘录

训练过程中可以观察到 loss 逐步下降。日志中部分训练进度如下：

```text
epoch 1 [train]:  10%|...| 99/1028  [02:19<18:34, 1.20s/it, loss=27.1289]
epoch 1 [train]:  15%|...| 156/1028 [03:35<18:06, 1.25s/it, loss=25.07]
epoch 1 [train]:  59%|...| 606/1028 [12:55<08:37, 1.23s/it, loss=16.6146]
```

这些中间日志说明：

- 训练过程没有出现崩溃、NaN 或维度错误。
- loss 从约 `27` 下降到约 `16`，后续 epoch 汇总为 `13.4402`。
- 模型参数确实在被更新，训练链路是有效的。
- 每个 batch 在 CPU 上大约需要 1.2 到 1.5 秒，DenseNet 在 CPU 上训练速度较慢，这是预期现象。

注意：终端中出现的 tqdm 进度条重复、截断或覆盖现象，是命令行进度条刷新导致的显示问题，不代表训练逻辑异常。

## 6. 验证集结果

训练结束后得到如下验证集结果：

```text
[densenet] epoch 001/1 | train_loss=13.4402 | val_loss=8.2509 | val_mae=8.2509 | val_rmse=11.6640 | lr=1.00e-04 | time=1482.6s | gpu_mem_peak_mb=0.0  <- best
[main] training done. best val MAE = 8.2509
```

各指标含义如下：

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| `train_loss` | 13.4402 | 训练集平均 L1 loss，表示训练集上预测年龄与真实年龄平均相差约 13.44 岁。 |
| `val_loss` | 8.2509 | 验证集平均 L1 loss。由于本次 loss 是 L1，因此与 MAE 基本一致。 |
| `val_mae` | 8.2509 | 验证集平均绝对误差，表示验证集上预测年龄平均误差约 8.25 岁。 |
| `val_rmse` | 11.6640 | 验证集均方根误差，对大误差更敏感。 |
| `lr` | 1.00e-04 | 当前学习率。只训练 1 个 epoch，尚未触发 step scheduler 的学习率衰减。 |
| `time` | 1482.6s | 训练 1 个 epoch 用时约 24.7 分钟。 |
| `gpu_mem_peak_mb` | 0.0 | 当前使用 CPU，因此 GPU 显存峰值为 0。 |
| `<- best` | - | 当前 epoch 是验证集 MAE 最好的 epoch，因此保存为 best checkpoint。 |

`val_rmse=11.6640` 明显高于 `val_mae=8.2509`，说明验证集中可能存在一部分误差较大的样本。RMSE 会对这些大误差样本给予更高惩罚，因此数值会被拉高。

## 7. 测试集结果

测试阶段加载了训练过程中保存的 best checkpoint：

```text
[densenet] TEST loss=9.5263 | MAE=9.5263 | RMSE=13.7923  (from checkpoints_smoke\densenet_best.pth)
[main] test done. summary: {
  'test_loss': 9.526289731595586,
  'test_mae': 9.526288986206055,
  'test_rmse': 13.792280077858546,
  'n_test': 1978,
  'ckpt_path': 'checkpoints_smoke\\densenet_best.pth'
}
```

测试集指标含义如下：

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| `test_loss` | 9.5263 | 测试集 L1 loss。由于 loss 是 L1，因此与 MAE 基本一致。 |
| `test_mae` | 9.5263 | 测试集平均绝对误差，表示测试集上预测年龄平均误差约 9.53 岁。 |
| `test_rmse` | 13.7923 | 测试集均方根误差，对大误差更敏感。 |
| `n_test` | 1978 | 测试集样本数量。 |
| `ckpt_path` | checkpoints_smoke\densenet_best.pth | 测试阶段加载的 best checkpoint 路径。 |

测试集 MAE 高于验证集 MAE：

```text
val_mae  = 8.2509
test_mae = 9.5263
```

这属于正常现象，可能原因包括：

- 测试集样本分布比验证集更难。
- 当前只训练了 1 个 epoch，模型尚未充分收敛。
- batch size 较小，CPU 环境下 smoke test 更偏向流程验证，而不是性能调优。

## 8. 本次验证结论

本次 CPU smoke test 的验证效果良好。

从工程链路角度看，以下关键步骤均已通过：

- 数据集路径可访问
- train/val/test 数据划分正常
- DataLoader 能够正常读取图像和标签
- DenseNet-121 模型能够正常构建
- 预训练配置能够生效
- 训练过程能够完成前向传播和反向传播
- L1 loss 能够正常下降
- 验证集评估能够正常执行
- best checkpoint 能够正常保存
- 测试阶段能够从 checkpoint 加载模型
- 测试集指标能够正常输出

从数值角度看：

- 训练 loss 从中间日志中的约 `27` 逐步下降，最终 epoch 平均 `train_loss=13.4402`，说明模型在学习。
- 验证集 `val_mae=8.2509`，测试集 `test_mae=9.5263`，结果没有出现异常偏大、NaN 或不可解释的数值。
- `RMSE` 大于 `MAE`，符合年龄估计任务中存在部分大误差样本时的常见表现。

因此，本次结果可以作为“CPU 环境下完整实验代码链路通过”的验证依据。

## 9. 注意事项

本次结果不应被视为 DenseNet 模型的最终性能结论，原因如下：

- 只训练了 1 个 epoch，模型尚未充分收敛。
- 使用 CPU 训练，速度较慢，实验效率有限。
- batch size 仅为 4，主要服务于快速链路验证。
- 未进行多次重复实验，不能评估随机性带来的波动。
- 未与其他模型或超参数配置进行对比。

后续正式实验建议：

- 切换到 GPU 环境运行。
- 增加训练 epoch 数，例如 50、100 或按项目配置执行。
- 根据显存情况适当增大 batch size。
- 记录并绘制 train/val loss 曲线。
- 对比不同模型的 `MAE` 和 `RMSE`。
- 保存完整实验配置、checkpoint 和日志，便于团队复现。

## 10. 总结

本次 CPU smoke test 成功完成了 DenseNet 年龄估计实验的完整流程。虽然只训练了 1 个 epoch，但训练、验证、保存 checkpoint、加载 checkpoint 和测试评估均正常执行，输出指标也符合预期。

最终测试集结果：

```text
test_loss = 9.5263
test_mae  = 9.5263
test_rmse = 13.7923
n_test    = 1978
```

综合判断：本次 CPU 验证通过，可以向团队说明当前实验代码链路具备继续开展正式训练和模型对比实验的基础。
