# age_estimation_one_click.ipynb 使用说明

本 notebook 是 APPA-REAL 年龄估计实验的一键运行与结果展示版本。当前版本的核心原则是：不在 notebook 中重写训练、测试、数据读取或模型构建逻辑，而是尽量保留原工程代码的功能边界。

## 目前的临时修改

1. 当前默认参数为：

   ```python
   RUN_EVALUATION_IF_DATA_AVAILABLE = False
   ```

   含义是：默认只展示 `results/` 中已有的实验结果，不重新运行测试集评估。这样 Run All 不会训练模型，不会更新模型参数，也不会覆盖已有 checkpoint。

2. notebook 已经移除个人电脑写死路径。默认会自动从当前目录向上寻找项目根目录，并优先使用：

   ```text
   data/appa-real-release/
   ```

   如果数据集放在其他位置，可以设置环境变量：

   ```powershell
   $env:APPA_REAL_ROOT="D:\your\path\appa-real-release"
   ```

3. notebook 不直接复制训练循环或评估算法。完整测试流程通过原始入口执行：

   ```powershell
   python main.py --model <model> --mode test --data_root <data_root> --ckpt_dir checkpoints_notebook --out_dir results/notebook_run
   ```

4. notebook 中自绘的 MAE/RMSE 小柱状图已改为英文标题，避免中文字体缺失导致 Matplotlib glyph warning。

## 后续要完成的任务

1. 准备 APPA-REAL 数据集，并保证目录结构如下：

   ```text
   data/appa-real-release/
   ├── train/
   ├── valid/
   ├── test/
   ├── gt_avg_train.csv
   ├── gt_avg_valid.csv
   └── gt_avg_test.csv
   ```

2. 确认 Jupyter 使用正确内核，例如：

   ```text
   Python (cv-age)
   ```

3. 如果需要重新验证完整测试集结果，将参数改为：

   ```python
   RUN_EVALUATION_IF_DATA_AVAILABLE = True
   ```

   此时 Run All 会调用 `main.py --mode test`，使用已有 checkpoint 在测试集上重新计算 MAE、RMSE，并生成 `results/notebook_run/`。

4. 如果不希望 notebook 自动安装缺失依赖，可以改为：

   ```python
   INSTALL_MISSING_DEPENDENCIES = False
   ```

5. 如果只想验证部分模型，可以修改：

   ```python
   MODELS = ["densenet", "vit", "vit_baseline"]
   ```

   例如只验证 DenseNet：

   ```python
   MODELS = ["densenet"]
   ```

## 目前 notebook 做了什么测试

在当前默认设置下，Run All 会执行：

1. 自动定位项目根目录。
2. 检查展示所需依赖，例如 `numpy`、`pandas`、`matplotlib`。
3. 自动尝试定位 APPA-REAL 数据集。
4. 自动定位备份权重目录 `checkpoints_backup/`。
5. 不重新训练模型，也不重新测试模型。
6. 读取 `results/` 中已有实验结果。
7. 汇总 DenseNet、ViT、ViT baseline 的测试指标。
8. 展示单模型结果图，例如预测散点图、MAE 曲线和 loss 曲线。
9. 展示 `results/comparison/` 中三个模型的性能对比表和对比图，包括：
   - `scatter_three_panel.png`
   - `pred_distribution.png`
   - `per_age_bar.png`
   - `mae_curves_overlay.png`
   - `loss_curves_overlay.png`
   - `budget_extension_mae_curves.png`
   - `budget_extension_baseline_per_age.png`

当前默认设置验证的是 notebook 展示流程与结果读取流程；如果要验证 checkpoint 加载和完整测试集推理，需要将 `RUN_EVALUATION_IF_DATA_AVAILABLE` 改为 `True` 后重新 Run All。
