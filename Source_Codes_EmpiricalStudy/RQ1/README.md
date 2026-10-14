# RQ1：不同双目方法生成的点云质量如何？

当前集合为 72 个场景 × 7 种双目方法，共 504 份评价。主要逐场景数据为 `data/development_pointclouds_20261012_v2/per_scene.csv`；`data/geometry` 保存几何评价，`data/pointclouds` 链接原始点云。

记录目标点数、重复点、非有限点、点到表面误差、近表面点比例、左右可见表面覆盖率、目标深度误差与目标来源比例。真值只用于评价。

复算汇总：`python3 ../publication/reproduce_tables.py --out 新目录`（在 RQ1 目录执行）。复核原始点云：`python3 code/analyze.py --label 新标签`。历史标签不覆盖。抓取结果分别见 RQ2 和 RQ3。

