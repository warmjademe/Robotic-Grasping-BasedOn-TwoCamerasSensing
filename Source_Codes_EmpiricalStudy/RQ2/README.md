# RQ2：固定 GraspGen 时，不同双目方法的抓取成功率如何？

比较 7 种双目方法，每种 72 个场景，在 FANUC + Isaac Sim、Franka + Isaac Sim、Franka + MuJoCo 中分别评价。默认摩擦与低摩擦条件分开统计；成功率分母包含全部场景。

- `data/development_graspgen_20261012`：候选与原 FANUC 规划阶段。
- `results/isaac_physics_20261013`：FANUC + Isaac Sim 的物理评价。
- `data/platform_comparison_20261014` 与 `results/platform_comparison_20261014`：Franka 的两个引擎。
- `code/analyze_physics.py --out 新目录`：从当前三个平台记录复算成功率。
- `code/analyze.py`：保留的候选和规划分析入口。

旧 `native_validation_20261012` 是历史 ROBOGUIDE 记录，仅用于来源追溯；不参与物理成功率。实际轨迹位于上一级两个物理试验目录。

