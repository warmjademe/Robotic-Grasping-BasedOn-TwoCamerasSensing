#!/usr/bin/env python3
"""Register completed inference evidence; never infer robot success from candidates."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'grasp_module_inventory.json'
report = json.loads((ROOT/'validation/new_methods_validation.json').read_text())
assert report['passed'] and len(report['rows']) == 4
data = json.loads(path.read_text())
snapshot = ROOT/'validation/source_snapshots/inventory_before_new_methods.json'
if not snapshot.exists():
    snapshot.write_bytes(path.read_bytes())
for method in ['gpd', 'graspnet_baseline']:
    m = next(m for m in data['methods'] if m['id'] == method)
    rows = {r['route']:r for r in report['rows'] if r['method'] == method}
    assert set(rows) == {'stereo_v2', 'multiview'}
    folder = 'baseline_' + method
    m.update(status='已运行', directory=folder, weights_present=True, inference_trials=2,
             native_trials=0, native_successes=None, priority='已接入共同候选入口',
             empirical_reason='主组候选；两条感知路线的共同输入/输出已验证',
             unified_adapter_verified=True, inference_contract_verified=True,
             full_unified_planning_verified=False,
             unified_adapter_scope='pointcloud_input_and_grasp_candidate_output_only',
             route_compatibility='pointcloud_input_output_verified',
             common_inference_smoke=dict(stereo_candidates=rows['stereo_v2']['candidates'],
                 multiview_candidates=rows['multiview']['candidates'],
                 stereo_width_valid=rows['stereo_v2']['width_valid_count'],
                 multiview_width_valid=rows['multiview']['width_valid_count'],
                 multiview_scene_scope='target_only', device='CPU' if method == 'gpd' else 'CUDA / RTX 4090',
                 empirical_model_comparison=False),
             evidence_files=[folder+'/README.md', folder+'/integration_summary.json',
                 'validation/new_methods_validation.json', 'validation/new_backend_results_review.json',
                 *[r['result_dir']+'/grasp_generation.json' for r in rows.values()]])
    if method == 'gpd':
        m.update(compatibility='作者 C++ 几何采样与 Eigen 15 通道 LeNet 分类器；CPU 实测，两路线接入。',
                 integration_notes='双目保留相机来源；融合点来源未知时估计可见性，未虚构真实观测。宽度是物体跨度，坐标按建模指尖平面对齐 Panda。',
                 limits='各返回排名前 200；分类器正类分数高于负类的候选：双目 0、多目 25。候选数不是可抓数量。多目可见性适配保留 7358/16342 点；仍待 Panda 碰撞和规划。')
    else:
        m.update(compatibility='作者网络和 CUDA 算子已运行；GCC 13 解决编译问题，9 项算子数值检查通过。',
                 integration_notes='双目原始 780 候选经目标筛选为空；多目返回 200，其中 179 个宽度有效、21 个超宽未裁剪。空结果保留，未放宽阈值。',
                 limits='权重来自固定哈希公开镜像，162 个张量严格加载；尚未确认与作者原文件字节一致。多目只有目标云，作者碰撞筛选不包含完整环境。',
                 official_checkpoint_verified=False,
                 checkpoint_provenance_status='third_party_mirror_official_file_byte_identity_unverified')
summary = data['summary']
summary.update(methods_with_model_inference_evidence=5,
               methods_with_verified_common_input_output=5,
               already_run=[m['name'] for m in data['methods'] if m['id'] in data['selection']['primary_ids']],
               recommended_new_core=[],
               inference_contract_evidence=['validation/module_validation.json','validation/new_methods_validation.json'],
               interface_regression_evidence='validation/new_backend_interface_review_huashuo.json')
for key in ['verified_model_route_pairs','nonempty_candidate_pairs','empty_candidate_pairs']:
    summary[key] = report[key]
data['selection']['newly_selected_status']='common_input_output_inference_verified_planning_pending'
data['updated_at'] = datetime.now(timezone.utc).isoformat()
data['latest_implementation_evidence'] = 'validation/new_methods_validation.json'
path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({'updated':str(path), 'verified_methods':5, 'verified_pairs':report['verified_model_route_pairs']}))
