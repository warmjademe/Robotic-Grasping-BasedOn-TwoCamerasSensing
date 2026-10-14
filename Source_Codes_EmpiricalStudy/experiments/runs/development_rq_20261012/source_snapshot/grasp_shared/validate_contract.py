"""Offline fault checks for coordinate, unit, provenance and gripper contracts."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
from contract import validate, load, transform_points


def main():
    root = Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('historical_contract',root/'validation/source_snapshots/before_provenance_fix/contract.py')
    historical=importlib.util.module_from_spec(spec);spec.loader.exec_module(historical)
    def saved_prediction(name):
        a,m=historical.load(root/'validation/inputs'/(name+'.npz'))
        # These audited saved predictions predate the declaration fields.
        # Add declarations in memory for this CPU regression, not to artifacts.
        m=dict(m,purpose='grasp_inference',geometry_source='perception_prediction')
        return validate(a,m)
    arrays, metadata = saved_prediction('stereo_v2')
    checks = {}
    T = arrays['T_world_reference']
    p = arrays['scene_points_m'][:500]
    restored = transform_points(transform_points(p, np.linalg.inv(T)), T)
    err = float(np.max(np.abs(restored-p)))
    assert err < 1e-12
    checks['world_reference_roundtrip_max_error_m'] = err
    for name, mutate in [
        ('wrong_units_rejected', lambda a,m:m.update(units='mm')),
        ('unknown_gripper_rejected', lambda a,m:m.update(gripper_id='unverified_custom')),
        ('missing_optical_convention_rejected', lambda a,m:m.pop('reference_convention')),
        ('reflection_rejected', lambda a,m:a['T_world_reference'].__setitem__((slice(None),0),-a['T_world_reference'][:,0])),
        ('invalid_point_origin_rejected', lambda a,m:a.update(point_camera_ids=np.full(len(a['scene_points_m']),999)))
    ]:
        a = {k:v.copy() for k,v in arrays.items()};m = copy.deepcopy(metadata)
        mutate(a,m)
        try:validate(a,m)
        except (ValueError, KeyError):checks[name] = True
        else:raise AssertionError(name)
    views = []
    for name,n in [('stereo_v2',2),('multiview',4)]:
        a,m = saved_prediction(name)
        assert len(a['camera_origins_m']) == n
        assert 'depth' not in a and 'K' not in a and 'valid' not in a
        assert not m['renderer_depth_used'] and not m['object_pose_used']
        views.append(dict(route=name,cameras=n,scene_points=len(a['scene_points_m']),
                          target_points=len(a['target_points_m']),scene_scope=m['scene_scope']))
    record = dict(passed=True,checks=checks,real_saved_predictions=views,
                  robot_operations_performed=False,
                  provenance_regression_note='Known saved model predictions receive explicit declarations in memory; historical NPZ/JSON files are unchanged',
                  initial_export_note='stereo.npz is preserved pre-check export with one origin; stereo_v2 includes both calibrated camera origins and is the tested input')
    (root/'validation/contract_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
