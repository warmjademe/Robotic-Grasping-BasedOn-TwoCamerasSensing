#!/usr/bin/env python3
"""MVSFormer++ on industrial four-view RGB, preserving calibrated metric scale."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree
from perception import GRASP, write
sys.path.insert(0,str(GRASP/'shared'))
from contract import save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--episodes',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.base/'upstream/repo'));sys.path.insert(0,str(a.base/'scripts'))
    sys.path.insert(0,str(a.base.parent/'_shared'))
    from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet
    from attention_compat import install
    import reconstruction_io as io
    names=['left','right','aux_left','aux_right'];io.NAMES=names
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(8)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();backend=install()
    ck=a.base/'models/model_best.pth';cfg=json.loads((a.base/'models/config.json').read_text())['arch']['args']
    cfg['vit_path']=str(a.base/'models/full_checkpoint_contains_vit')
    model=DINOv2MVSNet(cfg)
    state=torch.load(ck,map_location='cpu',weights_only=False)['state_dict']
    model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items() if 'pe_dict' not in k},strict=True)
    del state;model.cuda().eval()
    write(a.out/'model.json',dict(method='mvsformerplusplus',checkpoint_sha256=sha(ck),source_sha256=sha(__file__),
        config_sha256=sha(a.base/'models/config.json'),strict_state_dict_load=True,weights_retrained=False,
        attention_backend=backend,model_load_and_provenance_seconds=time.perf_counter()-started,
        input_size=[640,448],precision='BF16 autocast',runtime='CUDA',gpu=torch.cuda.get_device_name(),
        seed=0,fusion='2 supporting source views; <1px reprojection; <1% depth difference; confidence>=0.5'))
    for index,episode in enumerate(a.episodes):
        src,out=a.inputs/episode,a.out/episode;out.mkdir();started=time.perf_counter()
        task=json.loads((src/'task.json').read_text());assert task['split']=='development'
        cameras=json.loads((src/'cameras.json').read_text())['cameras']
        rgb={};Ks={};Ts={};normalized={}
        for n in names:
            image=cv2.cvtColor(cv2.imread(str(src/cameras[n]['image'])),cv2.COLOR_BGR2RGB)
            h,w=image.shape[:2];rgb[n]=cv2.resize(image,(640,448))
            K=np.asarray(cameras[n]['K'],np.float32);K[0]*=640/w;K[1]*=448/h
            Ks[n]=K;Ts[n]=np.asarray(cameras[n]['world_from_camera'],np.float32)
            normalized[n]=((rgb[n].astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)
        seconds=0.
        for n in names:
            order=[n]+[v for v in names if v!=n];proj={}
            im=torch.tensor(np.stack([normalized[v] for v in order]),device='cuda')[None]
            for stage,scale in [(1,8),(2,4),(3,2),(4,1)]:
                mats=[]
                for v in order:
                    K=Ks[v].copy();K[:2]/=scale;P=np.zeros((2,4,4),np.float32)
                    P[0]=np.linalg.inv(Ts[v]);P[1,:3,:3]=K;mats.append(P)
                proj['stage'+str(stage)]=torch.tensor(np.stack(mats),device='cuda')[None]
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(im,proj,torch.linspace(.09,1.2,192,device='cuda')[None],tmp=[5.,5.,5.,1.])
            torch.cuda.synchronize();seconds+=time.perf_counter()-t
            d=pred['refined_depth'].float().squeeze().cpu().numpy()
            conf=pred['photometric_confidence'].float().squeeze().cpu().numpy()
            np.savez_compressed(out/(n+'_prediction.npz'),depth=d,confidence=conf,
                valid=np.isfinite(d)&(d>.15)&(d<1.),target=np.ones(d.shape,bool),rgb=rgb[n],K=Ks[n],world_from_camera=Ts[n])
            del pred
        points,colors=io.fuse(out)
        workspace=(np.abs(points[:,0])<.45)&(np.abs(points[:,1])<.4)&(points[:,2]>-.02)&(points[:,2]<.3)
        points,colors=points[workspace],colors[workspace]
        lo=np.asarray(task['target_roi_world_m']['min']);lo[2]=.003
        hi=np.asarray(task['target_roi_world_m']['max']);target=points[((points>=lo)&(points<=hi)).all(1)]
        raw=len(target)
        if raw>=16:
            dist,_=cKDTree(target).query(target,k=16);local=dist[:,1:].mean(1)
            target=target[local<=local.mean()+2*local.std()]
        hashes={n:sha(src/n) for n in ['left.png','right.png','aux_left.png','aux_right.png','cameras.json','task.json']}
        np.savez_compressed(out/'cloud.npz',scene_points_m=points.astype(np.float32),target_points_m=target.astype(np.float32),scene_colors=colors)
        ready=min(len(points),len(target))>128
        if ready:
            save(out/'grasp_input.npz',dict(scene_points_m=points.astype(np.float32),target_points_m=target.astype(np.float32),
                scene_colors=colors,camera_origins_m=np.asarray([Ts[n][:3,3] for n in names]),T_world_reference=Ts['left'],
                point_camera_ids=np.full(len(points),-1,dtype=np.int32)),dict(schema_version=1,units='m',frame_id='dataset_world',
                input_hashes=hashes,purpose='grasp_inference',geometry_source='perception_prediction',renderer_depth_used=False,
                object_pose_used=False,reference_convention='opencv_optical_x_right_y_down_z_forward',scene_scope='workspace_roi',
                gripper_id='franka_panda',perception_route='multiview',has_normals=False,has_colors=True,
                gripper_geometry=dict(max_width_m=.08,closing_axis='+y',approach_axis='+z',tcp_frame='panda_hand'),
                target_selection='fixed world ROI; predicted height >3mm; kNN16 mean+2std'))
        record=dict(episode_id=episode,method='mvsformerplusplus',split='development',seed=0,input_sha256=hashes,
            inference_seconds=seconds,perception_pipeline_seconds=time.perf_counter()-started,first_inference_after_model_load=index==0,
            geometry_truth_used=False,object_pose_used=False,native_robot_execution_performed=False,
            scene_points=len(points),raw_target_points=raw,target_points=len(target),grasp_input_ready=ready)
        write(out/'perception.json',record);print(json.dumps(record),flush=True)


if __name__=='__main__':main()
