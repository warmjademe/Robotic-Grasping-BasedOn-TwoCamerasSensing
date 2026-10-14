"""Freeze selected development candidates; CAD is evaluation-only."""
from pathlib import Path
import csv, hashlib, json, os
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation

RUN = Path(__file__).resolve().parents[1]
STUDY = RUN.parent
BASE = Path(os.environ.get('GRASPGEN_RUNTIME', '/home/qyb/RESEARCH/baseline_graspgen'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=RUN/'inputs'; out.mkdir(exist_ok=True)
    union={}; sources={}
    for rel in ['RQ2/results/native_validation_20261012/per_scene_native.csv',
                'RQ3/results/foundationstereo_comparison_20261012/per_scene.csv']:
        sources[rel]=sha(STUDY/rel)
        for r in csv.DictReader((STUDY/rel).open()):
            key=(r['episode_id'],r['perception'],r['grasp'])
            if key in union:
                assert union[key]['plan_sha256']==r['plan_sha256']
            union[key]=r
    assert len(union)==792
    records=[]; grouped={}
    for key,r in sorted(union.items()):
        if r['path_found']!='True': continue
        p=STUDY/r['plan_source']; assert sha(p)==r['plan_sha256']
        plan=json.loads(p.read_text()); sel=plan['selected']
        scene_rel='datasets/industrial_workpieces_v1/evaluation/'+r['episode_id']+'/scene.json'
        scene=json.loads((STUDY/scene_rel).read_text())
        target=next(x for x in scene['instances'] if x['role']=='target')
        mesh_rel='datasets/industrial_workpieces_v1/assets/'+target['object_id']+'/mesh_m.ply'
        sources[r['plan_source']]=sha(p); sources[scene_rel]=sha(STUDY/scene_rel)
        sources[mesh_rel]=sha(STUDY/mesh_rel)
        # Saved hand_pose already uses Panda/USD closing Y, approach Z.
        H=np.linalg.inv(np.array(target['world_from_object']))@np.array(sel['hand_pose_dataset_m'])
        q=Rotation.from_matrix(H[:3,:3]).as_quat()
        gid='case_%04d'%len(records)
        entry=dict(position=H[:3,3].tolist(),orientation=dict(w=float(q[3]),xyz=q[:3].tolist()),
                   cspace_position={'panda_finger_joint1':sel['jaw_opening_m']/2},
                   pregrasp_cspace_position={'panda_finger_joint1':sel['jaw_opening_m']/2},
                   closing_command_m=sel['closing_command_m'],kind='prediction')
        grouped.setdefault(target['object_id'],{})[gid]=entry
        records.append(dict(case_id=gid,episode_id=r['episode_id'],perception=r['perception'],grasp=r['grasp'],
             object_id=target['object_id'],condition=r['condition'],plan_source=r['plan_source'],
             scene_source=scene_rel,mesh_source=mesh_rel,plan_sha256=r['plan_sha256'],
             old_geometric_success=r['grasp_success']=='True',
             old_failure=r.get('capture_failure_reason') or r.get('native_failure_reason'),
             selected_rank=sel['rank'],selected_score=sel['score'],
             hand_pose_dataset_m=sel['hand_pose_dataset_m'],jaw_opening_m=sel['jaw_opening_m'],
             closing_command_m=sel['closing_command_m']))
    assert len(records)==206
    def save_group(name,grasps,mesh):
        obj=out/(name+'.obj'); mesh.export(obj)
        data=dict(format='isaac_grasp',format_version='1.0',created_with='Frozen RQ2/RQ3 selected candidates',
          object_file=str(obj),object_scale=1.,gripper_file=str(BASE/'upstream/GraspDataGen/bots/franka_panda.usd'),
          gripper_frame_link='panda_hand',open_limit='upper',
          finger_colliders=['panda_rightfinger','panda_leftfinger'],grasps=grasps)
        (out/(name+'.yaml')).write_text(yaml.safe_dump(data,sort_keys=False))
    for obj,grasps in grouped.items():
        mesh=trimesh.load(STUDY/'datasets/industrial_workpieces_v1/assets'/obj/'mesh_m.ply',force='mesh')
        save_group(obj,grasps,mesh)
    cal={}
    for name,dx,opened in [('centered_box',0.,False),('far_negative',.2,False),('open_negative',0.,True)]:
        cal[name]=dict(position=[dx,0.,-.1015],orientation=dict(w=1.,xyz=[0.,0.,0.]),
          cspace_position={'panda_finger_joint1':.04},pregrasp_cspace_position={'panda_finger_joint1':.04},
          closing_command_m=.08 if opened else 0.,kind='calibration',expected=name=='centered_box')
    save_group('calibration_box',cal,trimesh.creation.box([.03,.03,.02]))
    manifest=dict(protocol='paired-selected-candidates-v1',total_pipeline_records=792,
        selected_candidates=len(records),old_successes=sum(r['old_geometric_success'] for r in records),
        source_sha256=sources,records=records,groups={k:len(v) for k,v in grouped.items()},
        scope='development selected candidates only; no candidate reselection or inference')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['records','source_sha256']},ensure_ascii=False))

if __name__=='__main__':main()
