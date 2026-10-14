#!/usr/bin/env python3
"""Read saved predictions and truth in a separate evaluation-only process."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from perception import write


def backproject(d, K, T):
    y,x = np.indices(d.shape)
    p = np.stack([(x-K[0,2])*d/K[0,0], (y-K[1,2])*d/K[1,1], d],-1)
    return p@T[:3,:3].T+T[:3,3]


def evaluate(dataset, episode, prediction, output=None):
    scene = json.loads((dataset/'evaluation'/episode/'scene.json').read_text())
    assert json.loads((dataset/'inputs'/episode/'task.json').read_text())['split']=='development'
    cams = json.loads((dataset/'inputs'/episode/'cameras.json').read_text())['cameras']
    gt = np.load(dataset/'evaluation'/episode/'truth.npz', allow_pickle=False)
    cloud = np.load(prediction/'cloud.npz', allow_pickle=False)['target_points_m']
    target = next(x for x in scene['instances'] if x['role']=='target')
    mesh = trimesh.load(dataset/'assets'/target['object_id']/'mesh_m.ply', force='mesh')
    mesh.apply_transform(np.asarray(target['world_from_object']))
    visible = []
    stereo_visible = []
    for name,cam in cams.items():
        depth, mask = gt[name+'_depth_m'], gt[name+'_instance_id']==target['instance_id']
        points=backproject(depth,np.asarray(cam['K']),np.asarray(cam['world_from_camera']))[mask & (depth>0)]
        visible.append(points)
        if name in ['left','right']:stereo_visible.append(points)
    visible = np.concatenate(visible)
    stereo_visible = np.concatenate(stereo_visible)
    result = dict(episode_id=episode, evaluation_only=True, ground_truth_used_only_here=True,
                  target_cloud_points=len(cloud), reference_visible_points=len(visible),
                  stereo_reference_visible_points=len(stereo_visible),primary_coverage_reference='left/right',
                  thresholds_mm=[1,2,5], native_robot_execution_performed=False)
    if len(cloud):
        distances=[]
        for start in range(0,len(cloud),2048):
            distances.append(trimesh.proximity.closest_point(mesh,cloud[start:start+2048])[1])
        distances=np.concatenate(distances)
        tree=cKDTree(cloud)
        coverage=tree.query(visible)[0]
        stereo_coverage=tree.query(stereo_visible)[0]
        result.update(surface_mae_mm=float(distances.mean()*1000),surface_p95_mm=float(np.quantile(distances,.95)*1000),
            surface_precision={str(t):float(np.mean(distances<=t/1000)) for t in [1,2,5]},
            four_view_visible_coverage={str(t):float(np.mean(coverage<=t/1000)) for t in [1,2,5]},
            stereo_visible_coverage={str(t):float(np.mean(stereo_coverage<=t/1000)) if len(stereo_visible) else None for t in [1,2,5]})
    else:
        result.update(surface_mae_mm=None,surface_p95_mm=None,
            surface_precision={str(t):0. for t in [1,2,5]},four_view_visible_coverage={str(t):0. for t in [1,2,5]},
            stereo_visible_coverage={str(t):0. if len(stereo_visible) else None for t in [1,2,5]})
    if (prediction/'prediction.npz').exists():
        pred=np.load(prediction/'prediction.npz',allow_pickle=False)
        target_mask=(gt['left_instance_id']==target['instance_id']) & (gt['left_depth_m']>0)
        valid=target_mask & pred['valid']
        err=np.abs(pred['depth'][valid]-gt['left_depth_m'][valid])*1000
        result['stereo_target_depth']=dict(gt_visible_pixels=int(target_mask.sum()),valid_pixels=int(valid.sum()),
            valid_fraction=float(valid.sum()/target_mask.sum()),mae_mm=float(err.mean()) if len(err) else None,
            p95_mm=float(np.quantile(err,.95)) if len(err) else None,
            within_5mm_fraction_including_invalid=float(np.count_nonzero(err<=5)/target_mask.sum()))
    destination = output if output is not None else prediction/'geometry_evaluation.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    write(destination,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--episode',required=True);p.add_argument('--prediction',type=Path,required=True)
    p.add_argument('--output', type=Path, help='Separate evaluation output; preserves historical prediction artifacts')
    a=p.parse_args();print(json.dumps(evaluate(a.dataset,a.episode,a.prediction,a.output)),flush=True)
