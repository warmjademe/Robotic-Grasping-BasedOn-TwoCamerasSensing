"""CPU check of real saved predictions and skew-aware projection geometry."""
import argparse
import json
from pathlib import Path
import tempfile
import numpy as np
from contract import load, sha
from export_legacy_cloud import export


def main():
    p=argparse.ArgumentParser()
    for key in ['stereo','multiview','cameras']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];rows=[]
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        for route,source,cameras,ref in [('stereo',a.stereo,None,None),('multiview',a.multiview,a.cameras,'front')]:
            output=tmp/(route+'.npz');export(source,output,cameras,ref)
            data,metadata=load(output)
            rows.append(dict(route=route,source=str(source),source_record_sha256=sha(source/'reconstruction.json'),
                             points=len(data['scene_points_m']),camera_count=len(data['camera_origins_m']),
                             explicit_prediction_provenance_passed=True,scene_scope=metadata['scene_scope']))
        fixture=tmp/'synthetic_projection_fixture';fixture.mkdir()
        K=np.array([[100.,7.,10.],[0.,110.,10.],[0.,0.,1.]])
        depth=np.full((20,20),1.5);valid=np.ones((20,20),bool)
        np.savez(fixture/'reconstruction.npz',points=np.ones((200,3)),depth=depth,K=K,valid=valid)
        (fixture/'reconstruction.json').write_text(json.dumps(dict(method='CPU geometry fixture only',geometry_truth_used=False,
                object_pose_used=False,reconstruction_sha256=sha(fixture/'reconstruction.npz'))))
        (fixture/'calibration.json').write_text(json.dumps(dict(world_from_left_camera=np.eye(4).tolist(),
                calibration='Parallel calibrated pinhole',baseline_m=.06)))
        export(fixture,tmp/'skew.npz');data,_=load(tmp/'skew.npz')
        xyz=data['scene_points_m'];projected=xyz@K.T;uv=projected[:,:2]/projected[:,2:3]
        yy,xx=np.indices(depth.shape);pixels=np.column_stack((xx.ravel(),yy.ravel()))
        error=float(abs(uv-pixels).max());assert error<1e-5
    record=dict(passed=True,real_prediction_exports=rows,synthetic_skew_projection_max_pixel_error=error,
                model_inference_run=False,robot_operations_performed=False,
                historical_inputs_modified=False,source_sha256=sha(root/'shared/export_legacy_cloud.py'))
    (root/'validation/legacy_export_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
