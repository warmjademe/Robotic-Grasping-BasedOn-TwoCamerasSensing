"""Verify the actual two-route x three-model GPU smoke-test artifacts."""
import json
import importlib.util
from pathlib import Path
import numpy as np
from contract import load, sha


def main():
    root=Path(__file__).resolve().parents[1]
    # The GPU smoke artifacts predate stricter provenance declarations. Verify
    # them against their preserved code rather than rewriting historical inputs.
    spec=importlib.util.spec_from_file_location('historical_contract',root/'validation/source_snapshots/before_provenance_fix/contract.py')
    historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
    rows=[]
    for route in ['stereo_v2','multiview']:
        input_path=root/'validation/inputs'/(route+'.npz')
        arrays,meta=historical.load(input_path)
        for method in ['6dof_graspnet','contact_graspnet','graspgen']:
            name=route+'_'+method
            if name=='multiview_contact_graspnet':name+='_empty_contract_fixed'
            out=root/'validation/results'/name
            record=json.loads((out/'grasp_generation.json').read_text())
            assert record['input_npz_sha256']==sha(input_path)
            assert record['input_json_sha256']==sha(input_path.with_suffix('.json'))
            assert record['output_sha256']==sha(out/'grasps.npz')
            for source,digest in record['source_sha256'].items():
                actual=root/'shared'/source
                if sha(actual)!=digest:
                    choices=[root/'validation/source_snapshots/before_provenance_fix'/source]
                    if source=='infer.py':choices.append(root/'validation/source_snapshots/infer_before_empty_fix.py')
                    actual=next(p for p in choices if p.exists() and sha(p)==digest)
                assert sha(actual)==digest,(source,digest)
            with np.load(out/'grasps.npz',allow_pickle=False) as data:
                poses=data['T_world_grasp'];width=data['jaw_width_m'];scores=data['score']
                assert poses.shape==(len(scores),4,4) and width.shape==scores.shape
                assert np.isfinite(poses).all() and np.isfinite(width).all()
                assert np.allclose(np.linalg.det(poses[:,:3,:3]),1,atol=1e-3)
                assert np.allclose(data['model_poses']@np.asarray(record['grasp_frame']['T_model_grasp']),poses)
            assert not record['native_robot_execution_performed']
            assert record['width_source']==('network_prediction' if method=='contact_graspnet' else 'fixed_gripper_geometry')
            rows.append(dict(route=route,method=method,input_scope=meta['scene_scope'],
                             candidates=len(scores),input_output_contract_passed=True,
                             status='candidates_generated' if len(scores) else 'no_candidates',
                             width_source=record['width_source'],result_dir=str(out.relative_to(root))))
    record=dict(passed=True,rows=rows,verified_model_route_pairs=len(rows),
                nonempty_candidate_pairs=sum(x['candidates']>0 for x in rows),
                empty_candidate_pairs=sum(x['candidates']==0 for x in rows),
                preserved_failed_attempt='validation/logs/multiview_contact_graspnet.log',
                empty_output_fix='Preserve valid empty result; no model/threshold changes',
                downstream_robot_validation_performed=False,
                gpu_source_revision='preserved pre-provenance-fix source snapshots',
                latest_interface_regression='validation/review_regression.json',
                historical_checkpoint_fingerprint_limit='Initial 6DOF logs fingerprint index only; GraspGen logs fingerprint YAML only. Current full weight inventory is supplementary, not retroactive proof of historical weight bytes.',
                downstream_pending=['width-aware planning','full-scene collision','new gripper TCP calibration'])
    (root/'validation/module_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
