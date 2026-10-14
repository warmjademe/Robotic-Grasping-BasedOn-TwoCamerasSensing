# 基于双目感知的工业工件抓取实证研究

当前研究使用 72 个场景、7 种双目方法、5 种抓取模型，组成 11 种配置。RQ1 评价点云质量；RQ2 固定 GraspGen 比较双目方法；RQ3 固定 FoundationStereo 比较抓取模型。

三个物理平台为 FANUC + Isaac Sim、Franka + Isaac Sim、Franka + MuJoCo。原始记录包括 504 份点云评价、792 组候选，以及两种摩擦条件下共 1,244 次完整机械臂物理执行。未生成候选或未找到路径的任务保留在成功率分母内。

当前入口：

- 场景集合：`datasets/active_dataset_72.json`。
- 复算三个研究问题：`python3 publication/reproduce_tables.py --out 新目录`。
- 原始点云分析：`RQ1/code/analyze.py --label 新标签`。
- FANUC 物理执行及轨迹：`physics_validation_20261013`。
- Franka 两个引擎：`platform_comparison_20261014`。

代码、日志与数据包发布于 https://github.com/warmjademe/Robotic-Grasping-BasedOn-TwoCamerasSensing 。公开仓库提供 SHA-256 清单、数据下载与复算脚本。

清理记录：`publication/cleanup_manifest.json`。已退役的 ROBOGUIDE 调度、旧 RQ4 和外部数据集适配代码移到项目外备份。历史数据及 `source_snapshot` 保留；这些目录和旧文件名用于来源追溯，不属于当前研究入口。`RQ4` 仅保留历史数据。实验启动开关保持关闭。
