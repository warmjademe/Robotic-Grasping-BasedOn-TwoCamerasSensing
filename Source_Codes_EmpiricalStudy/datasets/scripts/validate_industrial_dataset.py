#!/usr/bin/env python3
"""Independent integrity, mesh, split and calibrated stereo checks. No model run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_erosion


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def validate(root):
    manifest=json.loads((root/'manifest.json').read_text())
    objects=json.loads((root/'objects.json').read_text())
    episodes=json.loads((root/'episodes.json').read_text())
    cameras=json.loads((root/'calibration/cameras.json').read_text())
    by_id={o['object_id']:o for o in objects}
    assert len(by_id)==24 and len(episodes)==216 and manifest['rgb_image_count']==864
    for f,info in manifest['files'].items():
        p=root/f
        assert p.stat().st_size==info['bytes'] and digest(p)==info['sha256'],f
    for o in objects:
        a=trimesh.load(root/o['mesh'],force='mesh');b=trimesh.load(root/o['robotstudio_mesh'],force='mesh')
        assert a.is_watertight and a.is_winding_consistent and a.volume>0,o['object_id']
        assert np.allclose(a.extents,o['extents_m'],atol=1e-7)
        assert np.allclose(a.extents,b.extents/1000,atol=1e-7)
        assert abs(a.bounds[0,2])<1e-7 and max(a.extents[:2])<.08
    split_families={s:{o['family'] for o in objects if o['split']==s} for s in ['development','test']}
    assert not split_families['development']&split_families['test']
    L=np.array(cameras['cameras']['left']['world_from_camera'])
    R=np.array(cameras['cameras']['right']['world_from_camera'])
    relative=np.linalg.inv(L)@R
    assert np.allclose(relative[:3,:3],np.eye(3)) and np.allclose(relative[:3,3],[.06,0,0])
    for c in cameras['cameras'].values():
        T=np.array(c['world_from_camera']);rotation=T[:3,:3]
        assert np.allclose(T[3],[0,0,0,1]) and np.allclose(rotation.T@rotation,np.eye(3)) and np.isclose(np.linalg.det(rotation),1)
    projected_errors=[];rendered_depth_errors=[];minimum_pixels=10**9;pose_groups={}
    for e in episodes:
        assert e['split']==by_id[e['object_id']]['split']
        inp=root/e['input_dir'];gt=root/e['evaluation_dir']
        assert {p.name for p in inp.iterdir()}=={'left.png','right.png','aux_left.png','aux_right.png','cameras.json','task.json'}
        task=json.loads((inp/'task.json').read_text())
        assert task['renderer_depth_provided'] is False and task['object_pose_provided'] is False
        assert digest(inp/'cameras.json')==digest(root/'calibration/cameras.json')
        scene=json.loads((gt/'scene.json').read_text())
        assert scene['grasp_success'] is None and scene['abb_execution_status']=='not_run'
        assert all(by_id[i['object_id']]['split']==e['split'] for i in scene['instances'])
        T=np.array(scene['instances'][0]['world_from_object'])
        prior=pose_groups.setdefault(e['matched_pose_id'],T)
        assert np.array_equal(prior,T)
        with np.load(gt/'truth.npz',allow_pickle=False) as truth:
            for name,c in cameras['cameras'].items():
                rgb=np.array(Image.open(inp/c['image']));depth=truth[name+'_depth_m'];mask=truth[name+'_instance_id']
                assert rgb.shape==(480,640,3) and depth.shape==mask.shape==(480,640)
                assert np.isfinite(depth).all() and (depth>=0).all() and rgb.std()>2
                minimum_pixels=min(minimum_pixels,int((mask==1).sum()))
            # Independently derive pixel-to-left-3D -> right projection from saved calibration.
            d=truth['left_depth_m'];mask=binary_erosion(truth['left_instance_id']==1,iterations=2);v,u=np.where(mask)
            step=max(1,len(u)//64);u=u[::step];v=v[::step];z=d[v,u]
            K=np.array(cameras['cameras']['left']['K']);p=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z]
            world=p@L[:3,:3].T+L[:3,3];rp=(world-R[:3,3])@R[:3,:3]
            ur=rp[:,0]*K[0,0]/rp[:,2]+K[0,2];vr=rp[:,1]*K[1,1]/rp[:,2]+K[1,2]
            expected_disparity=K[0,0]*.06/z
            projected_errors.append(float(np.max(np.abs((u-ur)-expected_disparity))))
            assert np.max(np.abs(v-vr))<1e-5
            ui=np.rint(ur).astype(int);vi=np.rint(vr).astype(int)
            valid=(ui>=0)&(ui<640)&(vi>=0)&(vi<480)
            ui=ui[valid];vi=vi[valid];zr=rp[valid,2]
            visible=truth['right_instance_id'][vi,ui]==1
            assert visible.sum()>8,e['episode_id']
            rendered_depth_errors.extend(np.abs(truth['right_depth_m'][vi[visible],ui[visible]]-zr[visible]).tolist())
    assert max(projected_errors)<1e-4 and minimum_pixels>0
    assert np.quantile(rendered_depth_errors,.95)<.002
    return dict(passed=True,object_count=len(objects),episode_count=len(episodes),rgb_image_count=864,
        checked_manifest_files=len(manifest['files']),mesh_units_verified=True,watertight_meshes=24,
        input_truth_separation_verified=True,split_family_disjoint=True,
        matched_pose_groups=len(pose_groups),minimum_visible_target_pixels=minimum_pixels,
        max_stereo_projection_algebra_error_px=max(projected_errors),
        rendered_left_to_right_depth_consistency_p95_mm=float(np.quantile(rendered_depth_errors,.95)*1000),
        rendered_stereo_checked_points=len(rendered_depth_errors),
        projection_check_scope='calibration algebra and saved left/right renderer-depth reprojection; not learned depth accuracy or real-camera calibration',
        ABB_execution_count=0,grasp_success_rate=None,
        evaluator_status='ABB execution and success-evaluator validation pending')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
    report=validate(a.dataset.resolve())
    (a.dataset/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
