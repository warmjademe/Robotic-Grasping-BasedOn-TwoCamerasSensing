"""CPU regressions for review findings; mocked predictions are never benchmarks."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import numpy as np
import contract
import infer
from export_legacy_cloud import prediction_record


def main():
    root=Path(__file__).resolve().parents[1]
    old=root/'validation/inputs/stereo_v2.npz'
    with np.load(old,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    metadata=json.loads(old.with_suffix('.json').read_text())
    metadata.update(purpose='grasp_inference',geometry_source='perception_prediction')
    contract.validate(arrays,metadata)
    checks={}
    for name,changes in [
        ('evaluation_only_rejected',dict(purpose='evaluation_only')),
        ('renderer_depth_rejected',dict(renderer_depth_used=True)),
        ('object_pose_rejected',dict(object_pose_used=True)),
        ('nonprediction_geometry_rejected',dict(geometry_source='ground_truth')),
        ('absent_provenance_rejected',dict(geometry_source=None)),
    ]:
        m=copy.deepcopy(metadata);m.update(changes)
        try:contract.validate(arrays,m)
        except ValueError:checks[name]=True
        else:raise AssertionError(name)
    original_cwd=Path.cwd();original_argv=sys.argv[:];original_root=infer.ROOT
    old_backend=sys.modules.get('backends')
    try:
        with tempfile.TemporaryDirectory() as td:
            temp=Path(td);(temp/'baseline_runtime').mkdir()
            (temp/'baseline_runtime/.venv-tf').symlink_to(Path(sys.prefix))
            upstream=temp/'author_upstream';upstream.mkdir()
            contract.save(temp/'observation.npz',arrays,metadata)
            np.savez_compressed(temp/'reconstruction.npz',points=arrays['target_points_m'])
            source_record=dict(method='CPU regression fixture',geometry_truth_used=False,
                               object_pose_used=False,reconstruction_sha256=contract.sha(temp/'reconstruction.npz'))
            for name,changes in [
                ('legacy_renderer_depth_rejected',dict(renderer_depth_used_for_reconstruction=True)),
                ('legacy_evaluation_only_rejected',dict(purpose='evaluation_only')),
                ('legacy_geometry_truth_rejected',dict(geometry_truth_used=True)),
                ('legacy_object_pose_rejected',dict(object_pose_used=True)),
                ('legacy_wrong_hash_rejected',dict(reconstruction_sha256='bad')),
            ]:
                (temp/'reconstruction.json').write_text(json.dumps(dict(source_record,**changes)))
                try:prediction_record(temp)
                except ValueError:checks[name]=True
                else:raise AssertionError(name)
            (temp/'reconstruction.json').write_text(json.dumps(source_record))
            prediction_record(temp);checks['declared_legacy_prediction_accepted']=True
            def fake_sixdof(points,reference):
                os.chdir(upstream)
                return np.eye(4)[None],np.array([.1]),np.array([.08]),dict(test_only=True)
            sys.modules['backends']=types.SimpleNamespace(sixdof=fake_sixdof,contact=None,graspgen=None)
            infer.ROOT=temp;os.chdir(temp)
            sys.argv=['infer.py','--method','6dof_graspnet','--input','observation.npz','--out','relative_output']
            with contextlib.redirect_stdout(io.StringIO()):infer.main()
            result=json.loads((temp/'relative_output/grasp_generation.json').read_text())
            assert (temp/'relative_output/grasps.npz').is_file()
            assert not (upstream/'relative_output').exists()
            checks['relative_paths_survive_author_chdir']=True
            assert result['input_purpose']=='grasp_inference'
            assert result['input_geometry_source']=='perception_prediction'
            assert result['input_contains_renderer_depth'] is False
            assert result['input_contains_scene_object_pose'] is False
            checks['verified_provenance_propagated']=True
    finally:
        os.chdir(original_cwd);sys.argv=original_argv;infer.ROOT=original_root
        if old_backend is None:sys.modules.pop('backends',None)
        else:sys.modules['backends']=old_backend
    record=dict(passed=True,checks=checks,validation_kind='CPU synthetic contract regression only',
                model_inference_run=False,robot_operations_performed=False,
                provenance_limit='Checks explicit declarations and hashes, not automatic detection of arbitrary GT-derived points',
                source_sha256={p.name:contract.sha(p) for p in (root/'shared').glob('*.py')})
    (root/'validation/review_regression.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
