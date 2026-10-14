"""Audit saved development point clouds. No model inference or robot commands."""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

STUDY = Path(__file__).resolve().parents[2]
RUN = STUDY / 'experiments/runs/development_rq_20261012'
NAMES = {'stereobm': 'StereoBM', 'sgm': 'SGM', 'foundationstereo': 'FoundationStereo',
         'psmnet': 'PSMNet', 'raft_stereo': 'RAFT-Stereo', 'igev_stereo': 'IGEV-Stereo',
         'gwcnet': 'GwcNet'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, groups):
    output = []
    for group in sorted({tuple(r[k] for k in groups) for r in rows}):
        selected = [r for r in rows if tuple(r[k] for k in groups) == group]
        result = dict(zip(groups, group))
        result.update(scenes=len(selected), contract_passed=sum(r['contract_passed'] for r in selected),
                      empty_targets=sum(r['target_points'] == 0 for r in selected),
                      nonfinite_points=sum(r['nonfinite_points'] for r in selected))
        for field in ['scene_points', 'raw_target_points', 'target_points', 'retained_fraction',
                      'near_surface_2mm_points', 'target_nn_median_mm', 'target_voxels_1mm',
                      'surface_mae_mm', 'surface_p95_mm', 'precision_2mm', 'coverage_2mm',
                      'precision_5mm', 'coverage_5mm', 'target_depth_mae_mm',
                      'target_source_pixel_fraction', 'target_source_depth_mae_mm']:
            values = np.asarray([r[field] for r in selected], dtype=float)
            assert np.isfinite(values).all(), field
            for stat, value in [('mean', np.mean(values)), ('median', np.median(values)),
                                ('min', np.min(values)), ('max', np.max(values))]:
                result[field + '_' + stat] = float(value)
        # These use the evaluator's existing tolerances. They are descriptive,
        # not newly introduced pass/fail gates for selecting models or cases.
        for threshold in [2, 5, 10]:
            result[f'scenes_surface_mae_gt_{threshold}mm'] = sum(
                r['surface_mae_mm'] > threshold for r in selected)
        output.append(result)
    return output


def analyze(label):
    assert label and all(c.isalnum() or c in '_-' for c in label)
    control = json.loads((STUDY / 'experiments/task_control.json').read_text())
    assert control['launch_enabled'] is False, 'This audit is for the user-stopped experiment'
    data_dir = STUDY / 'RQ1/data' / label
    result_dir = STUDY / 'RQ1/results' / label
    assert not data_dir.exists() and not result_dir.exists(), 'Preserve previous audit outputs'
    data_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    sources = {}

    def checked(path, expected=None):
        value = digest(path)
        if expected is not None:
            assert value == expected, str(path)
        sources[str(path.relative_to(STUDY.parent))] = value
        return path

    def read(path, expected=None):
        return json.loads(checked(path, expected).read_text())

    manifest = read(RUN / 'manifest.json')
    index = read(RUN / 'prediction_index.json')
    assert len(index) == manifest['expected_perception_results'] == 504
    assert len({(x['method'], x['episode_id']) for x in index}) == 504
    episodes = {r['episode_id']: r for r in read(STUDY / 'datasets/industrial_workpieces_v1/episodes.json')}
    contract_path = STUDY.parent / 'Source_Codes_Grasping_Modules/shared/contract.py'
    checked(contract_path)
    spec = importlib.util.spec_from_file_location('pointcloud_contract', contract_path)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    rows = []
    evaluation_cache = {}
    for item in index:
        folder = STUDY / item['source']
        record = read(folder / 'perception.json', item['record_sha256'])
        checked(folder / 'prediction.npz', item['prediction_sha256'])
        checked(folder / 'cloud.npz', item['cloud_sha256'])
        checked(folder / 'grasp_input.npz', item['grasp_input_sha256'])
        checked(folder / 'grasp_input.json')
        arrays, metadata = contract.load(folder / 'grasp_input.npz')
        scene = np.asarray(arrays['scene_points_m'])
        target = np.asarray(arrays['target_points_m'])
        with np.load(folder / 'cloud.npz', allow_pickle=False) as raw:
            assert np.array_equal(raw['target_points_m'], target)
            assert np.array_equal(raw['scene_points_m'], scene)
        geometry = read(STUDY / 'RQ1/data/geometry' / item['method'] / (item['episode_id'] + '.json'))
        assert geometry['prediction_cloud_sha256'] == item['cloud_sha256']
        assert geometry['primary_coverage_reference'] == 'left/right'
        assert geometry['target_cloud_points'] == record['target_points'] == len(target)
        assert record['scene_points'] == len(scene)
        assert record['grasp_input_ready'] and metadata['units'] == 'm'
        assert not record['geometry_truth_used'] and not record['object_pose_used']
        # Evaluation only: recover the original left-image pixel of each saved
        # point. GT instance IDs are never supplied to perception/grasp models.
        episode = item['episode_id']
        if episode not in evaluation_cache:
            dataset = STUDY / 'datasets/industrial_workpieces_v1'
            camera = read(dataset / 'inputs' / episode / 'cameras.json')['cameras']['left']
            scene_truth = read(dataset / 'evaluation' / episode / 'scene.json')
            target_id = next(x['instance_id'] for x in scene_truth['instances'] if x['role'] == 'target')
            truth_path = checked(dataset / 'evaluation' / episode / 'truth.npz')
            with np.load(truth_path, allow_pickle=False) as truth:
                evaluation_cache[episode] = (camera, target_id, truth['left_instance_id'], truth['left_depth_m'])
        camera, target_id, instance_image, truth_depth = evaluation_cache[episode]
        transform = np.asarray(camera['world_from_camera'])
        intrinsic = np.asarray(camera['K'])
        optical = (target - transform[:3, 3]) @ transform[:3, :3]
        uv = optical[:, :2] / optical[:, 2:3]
        uv = uv * np.array([intrinsic[0, 0], intrinsic[1, 1]]) + intrinsic[:2, 2]
        pixels = np.rint(uv).astype(int)
        assert np.max(np.abs(uv - pixels)) < .01, 'Saved point does not map to an original pixel'
        u, v = pixels.T
        assert u.min() >= 0 and v.min() >= 0 and u.max() < instance_image.shape[1] and v.max() < instance_image.shape[0]
        from_target = instance_image[v, u] == target_id
        assert from_target.any()
        source_depth_error = np.abs(optical[from_target, 2] - truth_depth[v[from_target], u[from_target]]) * 1000
        unique = np.unique(target, axis=0)
        spacing = cKDTree(unique).query(unique, k=2)[0][:, 1] * 1000
        meta = episodes[item['episode_id']]
        assert meta['split'] == 'development'
        row = dict(method=item['method'], episode_id=item['episode_id'],
                   object_id=meta['object_id'], family=meta['family'], condition=meta['condition'],
                   split='development', contract_passed=True, scene_points=len(scene),
                   raw_target_points=record['raw_target_points'], target_points=len(target),
                   retained_fraction=len(target) / record['raw_target_points'],
                   unique_target_points=len(unique), exact_duplicate_points=len(target) - len(unique),
                   nonfinite_points=int((~np.isfinite(scene).all(axis=1)).sum() +
                                        (~np.isfinite(target).all(axis=1)).sum()),
                   target_nn_median_mm=float(np.median(spacing)), target_nn_p95_mm=float(np.quantile(spacing, .95)),
                   target_voxels_1mm=len(np.unique(np.floor(target / .001).astype(np.int64), axis=0)),
                   surface_mae_mm=geometry['surface_mae_mm'], surface_p95_mm=geometry['surface_p95_mm'],
                   precision_2mm=geometry['surface_precision']['2'],
                   precision_5mm=geometry['surface_precision']['5'],
                   coverage_2mm=geometry['stereo_visible_coverage']['2'],
                   coverage_5mm=geometry['stereo_visible_coverage']['5'],
                   target_depth_mae_mm=geometry['stereo_target_depth']['mae_mm'],
                   target_depth_valid_fraction=geometry['stereo_target_depth']['valid_fraction'],
                   target_source_pixel_points=int(from_target.sum()),
                   nontarget_source_pixel_points=int((~from_target).sum()),
                   target_source_pixel_fraction=float(from_target.mean()),
                   target_source_depth_mae_mm=float(source_depth_error.mean()),
                   near_surface_2mm_points=round(len(target) * geometry['surface_precision']['2']),
                   source=item['source'], cloud_sha256=item['cloud_sha256'],
                   grasp_input_sha256=item['grasp_input_sha256'])
        rows.append(row)
    methods = summarize(rows, ['method'])
    conditions = summarize(rows, ['method', 'condition'])
    objects = summarize(rows, ['method', 'object_id'])
    assert all(r['scenes'] == 72 for r in methods) and len(methods) == 7
    save_csv(data_dir / 'per_scene.csv', rows)
    save_csv(data_dir / 'by_method.csv', methods)
    save_csv(data_dir / 'by_condition.csv', conditions)
    save_csv(data_dir / 'by_object.csv', objects)
    worst = sorted(rows, key=lambda x: x['surface_mae_mm'], reverse=True)[:20]
    save_csv(data_dir / 'largest_surface_error.csv', worst)
    source_manifest = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           analysis_source_sha256=digest(Path(__file__)), input_files=sources,
                           evaluated_clouds=len(rows), all_contracts_passed=True,
                           models_rerun=False, robot_commands_sent=False, model_or_threshold_tuning=False,
                           pixel_membership_scope='Evaluation-only GT IDs; saved points projected back to original left-image pixels within 0.01 px',
                           scope='Development RGB stereo; equal weight per scene; coverage uses left/right only',
                           point_count_gate='Existing shared contract requires scene and target >128 finite points',
                           accuracy_gate='No accuracy acceptance threshold was pre-registered; report continuous metrics')
    (data_dir / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    shutil.copy2(__file__, result_dir / Path(__file__).name)
    table = ['| 方法 | 场景点数中位数 | 工件点数中位数（最少—最多） | 2 mm 内点占比 | 2 mm 可见表面覆盖 | 平均表面误差 mm |',
             '|---|---:|---:|---:|---:|---:|']
    for r in methods:
        table.append(f"| {NAMES[r['method']]} | {r['scene_points_median']:,.0f} | "
                     f"{r['target_points_median']:,.0f}（{r['target_points_min']:,.0f}—{r['target_points_max']:,.0f}） | "
                     f"{r['precision_2mm_mean']:.1%} | {r['coverage_2mm_mean']:.1%} | {r['surface_mae_mm_mean']:.2f} |")
    detail = ['| 方法 | 过滤后保留比例 | 邻点间距中位数 mm | 表面误差 >5 mm 场景 |', '|---|---:|---:|---:|']
    for r in methods:
        detail.append(f"| {NAMES[r['method']]} | {r['retained_fraction_mean']:.1%} | "
                      f"{r['target_nn_median_mm_median']:.3f} | {r['scenes_surface_mae_gt_5mm']}/72 |")
    membership = ['| 方法 | 工件点云中来自真实目标像素的比例 | 这些目标像素的深度误差 mm |', '|---|---:|---:|']
    for r in methods:
        membership.append(f"| {NAMES[r['method']]} | {r['target_source_pixel_fraction_mean']:.1%} | "
                          f"{r['target_source_depth_mae_mm_mean']:.2f} |")
    text = ('# 开发集点云输出与抓取输入可行性分析\n\n'
            '本分析只读取停止实验时已经保存的 504 份点云及几何评价，不运行模型或机械臂。'
            '每种方法对应相同的 72 个开发场景，正式测试数据未混入。\n\n'
            f"全部 504 份输入通过现有接口校验：米制坐标、非空有限点、场景和目标各超过 128 点、"
            f"来源哈希一致。工件点数最少 {min(r['target_points'] for r in rows):,}，"
            f"最多 {max(r['target_points'] for r in rows):,}。这说明数据可被抓取模块读取；"
            '是否足够准确需结合下面的误差与覆盖率判断。\n\n' + '\n'.join(table) + '\n\n'
            '场景点数包含工作区内的桌面及其他可见几何；工件点数是按输入区域筛选并过滤后的目标点数。'
            '点数是已保存数组的实际长度，不是插值估计。\n\n'
            '“2 mm 内点占比”表示目标点云中距离真实目标表面不超过 2 mm 的点所占比例；'
            '偏低意味着较多点偏离目标表面，可能包括深度误差及目标区域混入的其他表面。'
            '“2 mm 可见表面覆盖”表示左右相机真值可见表面采样点中，附近 2 mm 内能找到预测点的比例；'
            '它不是整个工件包含背面的完整面积覆盖率。两个比例均先按场景计算再等权平均。\n\n'
            '平均表面误差为预测目标点到真实目标网格的最近距离，再对 72 场景等权平均。'
            '它与目标像素的深度误差含义不同。所有真值仅用于离线评价。\n\n' + '\n'.join(detail) + '\n\n'
            '邻点间距按去除完全重复坐标后的目标点计算，再汇总各场景的中位数。'
            '它描述点的疏密，不能替代精度。>5 mm 场景数只是已有评价容差下的描述，未新增模型筛选门槛。\n\n'
            + '\n'.join(membership) + '\n\n'
            '上表将已保存的工件点投回左相机原像素，并在离线评价时用真值实例编号判断该像素是否属于目标工件。'
            '它与“距离真实表面 2 mm 以内”不同：来自目标像素的点仍可能有深度误差，来自其他像素的点也可能碰巧靠近目标。'
            '“这些目标像素的深度误差”仅统计已进入工件点云且真值属于目标的像素，不涵盖被过滤掉的目标像素。\n\n'
            '当前工件提取使用固定世界坐标 ROI、预测高度大于 3 mm 和 kNN16 离群点过滤，未使用实例分割。'
            '非目标像素进入工件点云可能来自区域内其他物体，也可能来自深度误差把桌面或背景移入目标区域。'
            '因此表面误差反映深度计算与工件点云提取的共同结果，不能把全部偏差归因于抓取模型。\n\n'
            '当前没有预先登记“满足某一误差和覆盖率即可抓取”的硬阈值，因此不把输出点云判为已经保证抓取。'
            '抓取还需要两侧接触区域的局部几何、有效夹法和机械臂运动验证。'
            '逐场景、条件、工件及误差最大的案例保存在对应 data 目录；所有分析输入有 SHA-256 记录。\n')
    (result_dir / '点云可行性分析.md').write_text(text)
    print(json.dumps(dict(data=str(data_dir), report=str(result_dir / '点云可行性分析.md'),
                          clouds=len(rows), methods=methods), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='development_pointclouds_20261012')
    analyze(parser.parse_args().label)
