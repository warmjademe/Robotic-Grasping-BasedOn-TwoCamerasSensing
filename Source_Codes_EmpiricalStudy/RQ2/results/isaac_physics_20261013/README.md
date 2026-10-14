# RQ2：Isaac Sim 物理复测

读取旧计划选中的候选和原关节路径，未重新运行感知或候选选择。旧 ROBOGUIDE 结果保持原样。

- `by_method.csv`：两种摩擦条件下的方法汇总，分别提供已选路径与全部 72 场景的分母。
- `per_candidate.csv`：逐候选指标、旧几何结果、源计划路径和哈希。两个摩擦条件分别列出。
- RQ2/RQ3 共享的 FoundationStereo + GraspGen 路径复用同一次物理执行，不能作为独立重复。

[完整协议与原始记录](../../../physics_validation_20261013/README.md) · [物理复测报告](../../../physics_validation_20261013/analysis/物理复测报告.md)
