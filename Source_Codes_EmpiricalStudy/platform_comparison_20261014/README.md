# Franka 双平台物理执行对照

新增 Isaac Sim 原生 Franka Panda 与 MuJoCo Menagerie Panda 两条执行链，保留历史 FANUC 结果。复用开发集 72 场景、792 个任务的原始点云与候选，对全部候选任务重新进行 Franka 规划。规划器不读取真实工件位姿或 CAD；真实几何仅用于搭建评价用物理场景。

## 运行与调度

- NAS 为主协调节点，代码和数据根目录为 `/home/qyb/RESEARCH/Robotic_Grasping/Source_Codes_EmpiricalStudy/platform_comparison_20261014`。
- 两台主机各有独立的 `/home/qyb/RESEARCH/baseline_graspgen/.venv-mujoco`；Isaac 使用 `.venv-sim`。NAS 的 Isaac 环境从华硕复制，并修正 Python 解释器软链接、补齐 libGLU。
- `plan_batch.py` 在 NAS 以四个 CPU 进程计算，BLAS/OMP 各限制一线程。`physics_batch.py` 在每次分配前读取 CPU、内存、磁盘、GPU 进程及共享锁，NAS 优先，其次华硕。原有训练不会被停止。
- MuJoCo 使用 CPU，可与本次 Isaac GPU 任务并行；发现其他训练占用时不在该主机追加任务。Isaac 与现有机器人项目共用 `physics_validation_20261013/gpu.lock`。
- 华硕原有训练占用 GPU 时，`calibrate_host_when_idle.py` 等待其自然结束，随后进行本机校验。校验通过后才允许参与完整平台调度。
- 主机分工写入 `ops/*_jobs_v2/`，每条任务、平台、摩擦条件只分配一次。基础设施错误保留原始文件并停止该协调器的新分配；修复后使用明确的尝试版本，禁止覆盖成功或失败的算法结果。

## 评价协议

完整协议见 `protocols/protocol.json`。两平台使用相同七关节路径、两个材料条件、物体质量、驱动力限制和动作时序。Franka 的固定工位平移为 `(0.5, 0, 0)` m；历史 FANUC 使用其自身工位。该对照包含机器人、控制器和碰撞资产变化，按平台比较解释。

正确夹持、夹空、夹爪不闭合和障碍阻挡是四类校准，均单独归档。最初 MuJoCo 校准发现运动结束时存在 5.11 mm 控制滞后；在主实验前，为两个新平台统一增加 1 s 接近后等待，保留 2 mm / 0.5 度到位阈值及失败校准。新版动作总长 20.5 s，历史 FANUC 为 19.5 s，不把仿真时间当成实际工站节拍。

目标依靠真实接触、摩擦和重力运动，不使用附着，也不在初始化后写入物体位姿。执行通过要求底部离台至少 100 mm 并保持 2 s、末端到位以及全程禁止接触检查通过。物理抬升、到位、碰撞和放回分别记录。

MuJoCo 保留 Menagerie 的碰撞形状和惯性，采用独立手指控制及机器人重力补偿。凹工件用 VHACD 分解；早期错误的三角面输入格式导致 qhull 失败，保留在初次尝试目录。修正后使用独立 `mujoco_collision_v2` 缓存，并保存各凸部件及来源哈希。两引擎的凸分解算法不完全相同。

主批次曾因 Trimesh 4.5.3 对安装块两个凸部件删除近退化三角面而中断。修复保留 Qhull 的全部三角面并按外法向定向，凸包顶点与体积保持一致；原失败日志和已完成结果保留，未产生结果的任务重新分配到新目录。恢复记录在 `ops/mesh_import_recovery_20261014`，对应恢复脚本只处理这两个已诊断批次。协调器遇到错误时停止新分配并等待其他在途批次结束，避免遗漏另一台主机的输出。

## 数据与结果

- `inputs/manifest.json`：2088 个冻结输入/候选报告哈希，以及 119 个既有结果文件哈希。
- `results/plans`：全部 792 个规划记录，保留无候选、未找到路径和超时。
- `results/{isaac,mujoco}/v2`：主物理结果、轨迹与接触记录。
- `calibration`：工程校准和资产接入诊断，不计入主实验。
- `analysis`：逐任务数据和即时进度。尚未执行的计划保持待完成，不记为抓取失败。
- `RQ2/data/platform_comparison_20261014` 和 `RQ3/data/platform_comparison_20261014`：按研究问题关联的逐任务数据；相应 `results` 子目录为分组表。
- `RQ4/data/platform_comparison_20261014`：规划计时和物理批次计时。批次墙钟、摊销成本和单次延迟分开，保留主机与并发数。

在 NAS 上运行 `python code/summarize.py` 更新当前结果；解释器使用上述 `.venv-mujoco/bin/python`。完整成功率仅在该组所有任务已取得结果后生成。历史原始结果仍在 `physics_validation_20261013`，不覆盖。

## 环境来源

- Isaac Sim 5.1.0.0，Isaac Lab 2.3.2.post1，原生 `FRANKA_PANDA_HIGH_PD_CFG`。
- MuJoCo 3.3.7，Menagerie 提交 `d7b481bac63d3d600ab404ee74f7474c5efe8a67`；许可随原始 Panda 资产保留。
- 官方资料：[Isaac Franka](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/core_api_tutorials/tutorial_core_adding_manipulator.html)、[MuJoCo 建模](https://mujoco.readthedocs.io/en/stable/modeling.html)、[Menagerie](https://github.com/google-deepmind/mujoco_menagerie)。
- NAS 首次启动的文件监视配额告警通过把当前 `fs.inotify.max_user_watches` 从 65536 增至 524288 处理；未修改系统启动配置。
