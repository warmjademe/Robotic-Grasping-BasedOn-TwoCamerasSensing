# Stereo perception and grasp generation for industrial manipulation

本仓库共享基于双目感知的工业工件抓取实证研究代码、配置、输入、模型输出及物理仿真记录。

当前数据集为 **72 个场景**：8 个工件、每件 3 个位姿、单件 / 邻件 / 料框三种条件，共 144 张左右 RGB 图像。工件为轴套和安装块两类。七种双目方法与五种抓取模型组成 11 种配置；共享的 FoundationStereo + GraspGen 配置只计算一次。

| 研究问题 | 固定条件与比较对象 | 数据 |
|---|---|---|
| RQ1：点云质量 | 七种双目方法 | 504 份深度与点云评价 |
| RQ2：双目方法对抓取成功率的影响 | 固定 GraspGen，比较七种双目方法 | 每个平台、材料条件 504 个任务 |
| RQ3：抓取模型对成功率的影响 | 固定 FoundationStereo，比较五种抓取模型 | 每个平台、材料条件 360 个任务 |

双目方法：StereoBM、OpenCV StereoSGBM、PSMNet、GwcNet、RAFT-Stereo、IGEV-Stereo、FoundationStereo。抓取模型：GPD、6-DOF GraspNet、Contact-GraspNet、GraspNet-Baseline、GraspGen。神经网络使用已有预训练权重。

三个完整机械臂物理执行平台为 **FANUC M-20iD/25 + Isaac Sim、Franka + Isaac Sim、Franka + MuJoCo**。FANUC 有 206 条入选计划，Franka 有 208 条；默认摩擦和较低摩擦分别执行，共 1,244 条完整机械臂执行记录。每种方法的成功率以全部预定场景为分母，包括没有候选或未找到路径的任务。模型输出候选不直接算作抓取成功。

## 快速复算结果（仅需 Python 3.9+）

```bash
git clone https://github.com/warmjademe/Robotic-Grasping-BasedOn-TwoCamerasSensing.git
cd Robotic-Grasping-BasedOn-TwoCamerasSensing
python3 tools/verify_artifacts.py --git-only
python3 Source_Codes_EmpiricalStudy/publication/reproduce_tables.py --out reproduced_tables
```

输出点云质量表、三个平台的成功数/分母、按场景条件分组的结果，以及输入 SHA-256。脚本检查任务唯一性、执行完整性和成功判据。输出目录必须是新目录，避免覆盖历史结果。

## 获取完整原始数据

Git 中保存代码、配置、输入图像、逐场景评价表、原始执行 JSON 和日志。较大的点云、抓取候选、深度数组及物理轨迹以分包形式保存在本仓库的 [Release](https://github.com/warmjademe/Robotic-Grasping-BasedOn-TwoCamerasSensing/releases/tag/study-72-scenes-v1)。

```bash
python3 tools/fetch_data.py
python3 tools/verify_artifacts.py
```

下载器验证数据包和每个文件的 SHA-256，再恢复原目录。已有文件内容不一致时停止；可使用 `--asset fanuc-physics.tar.gz` 等参数选择数据包。完整清单与大小见 [tools/artifacts.json](tools/artifacts.json)。数据包包含成功与失败记录。

## 目录

- `Source_Codes_EmpiricalStudy/datasets`：72 场景入口、左右图像、标定、评价真值及场景构建代码。
- `Source_Codes_EmpiricalStudy/RQ1`：点云质量分析代码和逐场景结果。
- `Source_Codes_EmpiricalStudy/RQ2`、`RQ3`：候选、规划与物理结果的研究问题视图。
- `Source_Codes_EmpiricalStudy/experiments`：感知、候选生成、FANUC 规划及冻结的原始批次。
- `physics_validation_20261013`：FANUC + Isaac Sim、独立夹爪 GraspDataGen 试验和校准记录（位于实证研究目录内）。
- `platform_comparison_20261014`：Franka 两个引擎的计划、协议、原始执行、失败诊断和校准记录。
- `Source_Codes_Grasping_Modules`：共同点云接口、模型适配器和兼容性源码。
- `tools/upstreams.json`：实际使用的上游源码地址和提交；`third_party_licenses` 保留相应许可。
- `Source_Codes_EmpiricalStudy/latex`：论文正文 `main.tex` / `main.pdf`、独立提纲、参考文献及数据生成的图表。

## 数据语义与复现层次

当前场景集合以 `datasets/active_dataset_72.json` 为准。历史文件名中的 `development` 保留以维持原始引用及校验值；原场景目录中的 `episodes.json` / `manifest.json` 保留旧索引作为来源证据，索引中额外的 144 个场景不属于本次发布。运行记录中的机器路径标识原执行位置。

两个旧 ROBOGUIDE 汇总 CSV 因被物理试验输入清单引用而保留，仅用于核对候选来源；其几何标签不参与本研究的物理抓取成功率。RQ4 已退出当前研究，历史计时只作为运行诊断。目录内 `source_snapshot` 是当时实际使用的源码证据，保留原样；当前入口说明优先于历史 README。

复算汇总表无需 GPU。重新检查点云和物理轨迹需下载数据并安装 NumPy、SciPy、trimesh、matplotlib。重新执行神经网络及仿真还需要作者模型权重、各模型的独立运行环境和相应模拟器；见 [docs/reproduction.md](docs/reproduction.md)。本次发布验证的是保存数据的复算与完整性，没有重新运行整套 GPU 实验。

## 许可与引用

上游代码、机器人资产和模型权重分别遵循其原有许可，本仓库不对它们重新授权。模型权重通过作者渠道获取，不随数据包重新分发。使用前请查看各上游的许可；Contact-GraspNet 等依赖可能限制用途。论文和方法引用见 `Source_Codes_EmpiricalStudy/latex/references.bib`。本研究自编代码及实验记录暂未授予统一的开源许可证。
