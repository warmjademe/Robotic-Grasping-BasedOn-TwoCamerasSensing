# RQ1 点云数据

- `pointclouds/<方法>/<场景>/`：原始 `prediction.npz`、`cloud.npz`、`grasp_input.npz` 与元数据；7 个相对符号链接指向共享原始批次，避免复制或改变来源路径。
- `geometry/`：504 份已保存的开发几何评价，保留原位置和哈希。
- `development_pointclouds_20261012_v2/per_scene.csv`：当前 RQ1 的 504 行逐点云指标。
- `development_pointclouds_20261012_v2/by_method.csv`、`by_condition.csv`、`by_object.csv`：按方法、场景条件及工件汇总。
- `development_pointclouds_20261012_v2/source_manifest.json`：分析所用 3,244 个输入文件的来源哈希。
- `development_pointclouds_20261012/`：初始数量与几何分析，作为历史版本保留；当前结果引用 v2。
- `perception_metrics.csv`：旧开发批次导出的深度／几何指标，保留以便交叉核对。
- `formal_perception.csv`：旧协议下 144 测试场景 × 7 方法的 1,008 条感知记录。与开发集分开保存，未用于本次 504 份点云结论。

候选、路径与原生执行文件移至 `../archive/pre_pointcloud_rq1_20261012/data/`。所有同场景多方法比较保留配对身份；方法重复、随机种子复用和点数量不增加独立工件样本数。
