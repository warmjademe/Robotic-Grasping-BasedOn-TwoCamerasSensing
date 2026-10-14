# RQ3：固定 FoundationStereo 时，不同抓取模型的成功率如何？

固定同一份 FoundationStereo 点云，比较 GPD、6-DOF GraspNet、Contact-GraspNet、GraspNet-Baseline、GraspGen。每种模型使用同一批 72 个场景，在 FANUC + Isaac Sim、Franka + Isaac Sim、Franka + MuJoCo 中评价，默认摩擦与低摩擦分开统计。

`data/development_foundationstereo_20261012` 保存候选与规划；`results/isaac_physics_20261013` 保存 FANUC 物理评价；`data/platform_comparison_20261014` 和对应 results 保存 Franka 两个引擎的结果。

运行 `python3 code/analyze.py --out 新目录` 可复算三个平台。旧 `foundationstereo_comparison_20261012/per_scene.csv` 被原始物理输入清单引用，因此保留其字节用于校验，旧几何成功标签不计入物理成功率。完整轨迹位于上一级两个物理试验目录。
