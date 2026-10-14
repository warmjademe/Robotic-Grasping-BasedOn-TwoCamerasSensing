#!/usr/bin/env python3
"""Audit saved real shared-CLI outputs without rerunning models or robot I/O."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from contract import load, sha
from infer import validate_pose_batch

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for method in ['gpd', 'graspnet_baseline']:
        for route in ['stereo_v2', 'multiview']:
            inp = ROOT/'validation/inputs_20261011_new_methods'/f'{route}.npz'
            _, meta = load(inp)
            out = ROOT/'validation/results_20261011_new_methods'/f'{method}_{route}'
            record = json.loads((out/'grasp_generation.json').read_text())
            with np.load(out/'grasps.npz', allow_pickle=False) as data:
                a = {k: data[k] for k in data.files}
            n = record['returned']
            validate_pose_batch(a['model_poses'], n, 'author')
            validate_pose_batch(a['T_world_grasp'], n, 'common')
            assert np.allclose(a['model_poses'] @ a['T_model_grasp'], a['T_world_grasp'])
            assert a['jaw_width_m'].shape == a['score'].shape == (n,)
            assert np.isfinite(a['jaw_width_m']).all() and np.isfinite(a['score']).all()
            expected_valid = (a['jaw_width_m'] > 0) & (a['jaw_width_m'] <= .080001)
            assert np.array_equal(a['gripper_width_valid'], expected_valid)
            assert record['width_valid_count'] == int(expected_valid.sum())
            assert record['input_npz_sha256'] == sha(inp)
            assert record['input_json_sha256'] == sha(inp.with_suffix('.json'))
            assert record['output_sha256'] == sha(out/'grasps.npz')
            for f, digest in record['source_sha256'].items():
                assert digest == sha(ROOT/'shared'/f)
            assert record['inference_completed'] is True
            assert record['status'] == ('candidates_generated' if n else 'no_candidates')
            assert record['scene_scope'] == meta['scene_scope']
            assert record['native_robot_execution_performed'] is False
            assert record['physical_robot_used'] is False
            assert record['common_grasp_origin'] == 'panda_hand'
            assert record['geometry_alignment']
            rows.append(dict(method=method, route=route, candidates=n,
                             width_valid_count=int(expected_valid.sum()),
                             width_source=record['width_source'], input_scope=meta['scene_scope'],
                             input_output_contract_passed=True, status=record['status'],
                             input_npz_sha256=sha(inp), output_sha256=record['output_sha256'],
                             result_dir=str(out.relative_to(ROOT))))
    historical = json.loads((ROOT/'validation/module_validation.json').read_text())
    assert historical['passed']
    combined = historical['rows'] + rows
    report = dict(passed=True, checked_at=datetime.now(timezone.utc).isoformat(), rows=rows,
                  new_verified_model_route_pairs=len(rows),
                  verified_model_route_pairs=len(combined),
                  nonempty_candidate_pairs=sum(r['candidates'] > 0 for r in combined),
                  empty_candidate_pairs=sum(r['candidates'] == 0 for r in combined),
                  historical_evidence='validation/module_validation.json',
                  input_comparison='validation/inputs_20261011_new_methods/validation.json',
                  current_source_hashes={f:sha(ROOT/'shared'/f) for f in ['infer.py','backends.py','contract.py']},
                  full_unified_planning_verified=False, physical_robot_used=False,
                  scope='Real model inference and common input/output integration; not a grasp-success benchmark',
                  checkpoint_limitation='GraspNet-Baseline mirror strictly restores but author-byte identity is unverified')
    (ROOT/'validation/new_methods_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
