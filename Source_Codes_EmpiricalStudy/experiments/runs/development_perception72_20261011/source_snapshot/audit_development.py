#!/usr/bin/env python3
"""Verify actual pilot artifacts and resolved dependency recovery without reruns."""
import json
from pathlib import Path
import numpy as np
from run_development import HERE, DATA, GRASP, sha, write

run=HERE/'runs/development_pilot_20261011'
recovery=HERE/'runs/development_raft_dependency_recovery_20261011'
status=json.loads((run/'status.json').read_text())
assert status['status']=='development_inference_finished_abb_pending'
assert json.loads((recovery/'status.json').read_text())['status']=='finished'
geometry=[];grasp_rows=[];no_input=[];predictions=[]
for method in status['perception_methods']:
    origin=recovery if method=='raft_stereo' else run
    model=json.loads((origin/'perception'/method/'model.json').read_text())
    for ep in status['episodes']:
        path=origin/'perception'/method/ep
        info=json.loads((path/'perception.json').read_text())
        assert info['geometry_truth_used'] is False and info['object_pose_used'] is False
        for name,digest in info['input_sha256'].items():assert sha(DATA/'inputs'/ep/name)==digest
        arrays=np.load(path/'cloud.npz',allow_pickle=False)
        assert len(arrays['target_points_m'])==info['target_points']
        assert np.isfinite(arrays['target_points_m']).all()
        geometry.append(dict(method=method,**json.loads((path/'geometry_evaluation.json').read_text())))
        predictions.append(info)
        models=status['grasp_methods'] if method in status['perceptions_for_RQ2'] else ['graspgen']
        for backend in models:
            output=origin/'grasps'/method/ep/backend
            if not info['grasp_input_ready']:
                assert not output.exists()
                no_input.append(dict(perception=method,episode=ep,grasp=backend,reason='insufficient_predicted_target_points'))
                continue
            record=json.loads((output/'grasp_generation.json').read_text())
            assert record['input_npz_sha256']==sha(path/'grasp_input.npz')
            assert record['input_json_sha256']==sha(path/'grasp_input.json')
            assert record['output_sha256']==sha(output/'grasps.npz')
            assert record['native_robot_execution_performed'] is False
            a=np.load(output/'grasps.npz',allow_pickle=False)
            count=record['returned'];T=a['T_world_grasp']
            assert T.shape==(count,4,4) and np.isfinite(T).all()
            if count:
                assert np.allclose(T[:,:3,:3].transpose(0,2,1)@T[:,:3,:3],np.eye(3),atol=1e-3)
                assert np.allclose(np.linalg.det(T[:,:3,:3]),1,atol=1e-3)
            assert int(a['gripper_width_valid'].sum())==record['width_valid_count']
            grasp_rows.append(dict(perception=method,episode=ep,grasp=backend,
                candidates=count,width_valid=record['width_valid_count'],status=record['status']))
assert len(predictions)==48 and len(grasp_rows)+len(no_input)==96
report=dict(passed=True,scope='artifact consistency, not ABB execution or grasp success',
    perception_completed=len(predictions),planned_grasp_jobs=96,grasp_inference_completed=len(grasp_rows),
    grasp_inference_with_candidates=sum(r['candidates']>0 for r in grasp_rows),
    grasp_inference_empty=sum(r['candidates']==0 for r in grasp_rows),
    no_perception_input=len(no_input),native_abb_executions=0,abb_grasp_success_rate=None,
    test_set_started=False,development_episode_count=len(status['episodes']),
    original_dependency_failure_preserved=True,dependency_recovery='completed',
    grasp_rows=grasp_rows,no_input_rows=no_input,geometry_rows=geometry)
write(run/'artifact_audit.json',report)
print(json.dumps({k:v for k,v in report.items() if not k.endswith('_rows')}))
