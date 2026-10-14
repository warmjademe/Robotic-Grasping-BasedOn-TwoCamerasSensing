#!/usr/bin/env python3
"""Independently audit saved new-backend inference evidence, never run a model."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from contract import load, sha
from infer import validate_pose_batch

ROOT = Path(__file__).resolve().parents[1]


def import_backend(method):
    path = ROOT/('baseline_'+method)/'backend.py'
    spec = importlib.util.spec_from_file_location('review_'+method, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(method, route):
    base = ROOT/('baseline_'+method)
    out = ROOT/'validation/results_20261011_new_methods'/(method+'_'+route)
    record = json.loads((out/'grasp_generation.json').read_text())
    input_path = ROOT/'validation/inputs_20261011_new_methods'/(route+'.npz')
    arrays, meta = load(input_path)
    assert record['input_npz_sha256'] == sha(input_path)
    assert record['input_json_sha256'] == sha(input_path.with_suffix('.json'))
    assert record['output_sha256'] == sha(out/'grasps.npz')
    with np.load(out/'grasps.npz', allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    widths, scores = result['jaw_width_m'], result['score']
    author = validate_pose_batch(result['model_poses'], len(scores), 'saved author poses')
    common = validate_pose_batch(result['T_world_grasp'], len(scores), 'saved common poses')
    transform = result.get('T_model_to_common_grasp', result.get('T_model_grasp'))
    if transform is None:
        transform = np.asarray(record['T_model_to_common_grasp'])
    transform = np.asarray(transform)
    if transform.shape == (4, 4):
        validate_pose_batch(transform[None], 1, 'saved adapter')
    else:
        transform = validate_pose_batch(transform, len(scores), 'saved per-candidate adapters')
    assert np.allclose(author @ transform, common, atol=1e-12)
    assert widths.shape == scores.shape == (record['returned'],)
    assert np.isfinite(widths).all() and np.isfinite(scores).all()
    expected_valid = (widths > 0) & (widths <= .08+1e-6)
    assert np.array_equal(result['gripper_width_valid'], expected_valid)
    assert record['width_valid_count'] == int(expected_valid.sum())
    assert record.get('native_robot_execution_performed') is False
    assert record.get('physical_robot_used') is False
    details = {}
    backend = import_backend(method)
    if method == 'graspnet_baseline':
        raw_path = out/'raw_extra.npz'
        assert sha(raw_path) == record['raw_extra_sha256']
        with np.load(raw_path, allow_pickle=False) as raw:
            rows = raw['raw_grasp_rows']
            selected = raw['selected_raw_indices']
            target_mask = raw['target_mask']
            collision = raw['collision_mask']
            valid_ids = np.flatnonzero(target_mask & ~collision)
            expected_ids = valid_ids[np.argsort(-rows[valid_ids, 0], kind='stable')[:record['candidate_budget']]]
            assert np.array_equal(expected_ids, selected)
            assert np.array_equal(rows[selected, 0], scores)
            assert np.array_equal(rows[selected, 1], widths)
            poses = np.tile(np.eye(4), (len(selected), 1, 1))
            poses[:, :3, :3] = rows[selected, 4:13].reshape(-1, 3, 3)
            poses[:, :3, 3] = rows[selected, 13:16]
            assert np.array_equal(arrays['T_world_reference'][None] @ poses, author)
            adapter, evidence = backend.model_to_common(rows[selected, 3])
            assert np.array_equal(adapter, transform)
            assert np.array_equal(raw['decoded_jaw_width_m'], widths)
            from scipy.spatial import cKDTree
            T = arrays['T_world_reference']
            target_ref = ((arrays['target_points_m'].astype(np.float32)-T[:3, 3]) @ T[:3, :3]).astype(np.float32)
            distances = cKDTree(target_ref).query(rows[:, 13:16])[0]
            assert np.allclose(distances, raw['target_distances_m'], atol=1e-12)
            assert np.array_equal(distances < record['parameters']['target_distance_m'], target_mask)
            assert record['raw_candidates'] == len(rows)
            assert record['target_filtered_candidates'] == int(target_mask.sum())
            assert record['collision_rejected'] == int(collision.sum())
            details.update(raw_candidates=len(rows), target_candidates=int(target_mask.sum()),
                           collision_rejected=int(collision.sum()), candidate_order_verified=True,
                           target_distances_independently_recomputed=True,
                           returned_widths_equal_author_decoded_values=True)
        files = {'backend.py': base/'backend.py', 'graspnet.py': base/'upstream/graspnet-baseline/models/graspnet.py',
                 'collision_detector.py': base/'upstream/graspnet-baseline/utils/collision_detector.py'}
        for name, digest in record['backend_dependency_sha256'].items():
            assert sha(files[name]) == digest
        checkpoint = base/'models'/Path(record['checkpoint']).name
        assert sha(checkpoint) == record['checkpoint_sha256'] == record['checkpoint_provenance']['sha256']
        assert record['strict_checkpoint_restore'] is True
        details.update(checkpoint_sha256=sha(checkpoint), strict_checkpoint_restore=True,
                       official_checkpoint_verified=record['official_checkpoint_verified'],
                       weight_provenance=record['checkpoint_provenance'])
    elif method == 'gpd':
        backend.verify_author_files()
        assert np.array_equal(backend.model_to_common(), transform)
        with np.load(out/'author_output.npz', allow_pickle=False) as raw:
            keep = raw['retained']
            assert np.array_equal(raw['T_world_author'][keep], author)
            assert np.array_equal(raw['scores'][keep], scores)
            assert np.array_equal(raw['widths_m'][keep], widths)
            details['raw_candidates'] = len(keep)
        for name, digest in record['checkpoint_files_sha256'].items():
            assert sha(base/name) == digest
        details.update(official_preflight_passed=True, width_source=record['width_source'],
                       width_semantics=record['width_semantics'],
                       positive_author_logits=int((scores > 0).sum()),
                       score_definition=record['score_semantics'],
                       visibility_adapter=record['visibility_adapter'],
                       author_nominal_opening_m=record['author_nominal_max_opening_m'])
    else:
        raise ValueError(method)
    return dict(method=method, route=route, passed=True, returned=len(scores), width_valid=int(expected_valid.sum()),
                width_invalid=int((~expected_valid).sum()), status='candidates_generated' if len(scores) else 'no_candidates',
                scene_scope=meta['scene_scope'], input_npz_sha256=sha(input_path),
                output_sha256=sha(out/'grasps.npz'), record_sha256=sha(out/'grasp_generation.json'), details=details,
                learned_model_reexecuted=False, native_robot_execution_performed=False,
                evidence=str(out.relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['graspnet_baseline'], choices=['gpd', 'graspnet_baseline'])
    parser.add_argument('--out', type=Path, default=ROOT/'validation/new_backend_results_review.json')
    args = parser.parse_args()
    rows = [audit(method, route) for method in args.methods for route in ['stereo_v2', 'multiview']]
    report = dict(passed=True, rows=rows, pair_count=len(rows), checked_methods=args.methods,
                  audit_script_sha256=sha(__file__),
                  scope='Read-only independent audit of saved model results; new inference and robot actions were not performed',
                  actual_robot_tcp_calibrated=False, panda_mesh_collision_verified=False,
                  limitation='Candidate generation and modeled distal-tip alignment do not measure grasp success; target-only inputs cannot establish full-scene collision safety')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(passed=True, pairs=len(rows), out=str(args.out))))


if __name__ == '__main__':
    main()
