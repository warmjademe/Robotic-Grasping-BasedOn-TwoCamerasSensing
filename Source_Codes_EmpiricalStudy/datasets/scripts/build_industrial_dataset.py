#!/usr/bin/env python3
"""Generate original industrial meshes and calibrated RGB episodes, not grasp labels.

Inference folders contain RGB/calibration/task instructions only. All object poses,
depth, masks and robot-import manifests are separated in evaluation/ and assets/.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, Point

SEED = 20261011
WIDTH, HEIGHT, FOCAL, BASELINE = 640, 480, 700., .06
FAMILIES = ['spacer', 'mounting_block', 'hex_nut', 'flange_bushing', 'l_plate', 'u_channel']
NAMES = ['轴套／隔套', '安装块', '六角螺母形件', '法兰轴套', 'L形连接板', 'U形槽块']
DEV_FAMILIES = {'spacer', 'mounting_block'}
CONDITIONS = ['single', 'neighbours', 'bin_walls']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ring(outer, inner, height, sides=64):
    angles = np.arange(sides)*2*np.pi/sides
    poly = Polygon(np.c_[np.cos(angles)*outer, np.sin(angles)*outer],
                   [np.c_[np.cos(np.arange(64)*2*np.pi/64)*inner,
                          np.sin(np.arange(64)*2*np.pi/64)*inner]])
    return trimesh.creation.extrude_polygon(poly, height, engine='earcut')


def make_part(family, variant):
    s = [.85, 1., 1.12, 1.22][variant]
    if family == 'spacer':
        mesh = ring(.020*s, .010*s, .024*s)
    elif family == 'hex_nut':
        mesh = ring(.025*s, .010*s, .014*s, sides=6)
    elif family == 'flange_bushing':
        profile = np.array([[.009,0],[.027,0],[.027,.007],[.017,.007],[.017,.032],[.009,.032],[.009,0]])*s
        mesh = trimesh.creation.revolve(profile, sections=64)
    elif family == 'mounting_block':
        poly = Polygon([[-.025,-.019],[.025,-.019],[.025,.019],[-.025,.019]])
        poly = poly.difference(Point(-.012,0).buffer(.004, resolution=24)).difference(Point(.012,0).buffer(.004, resolution=24))
        mesh = trimesh.creation.extrude_polygon(poly, .017, engine='earcut')
        mesh.apply_scale(s)
    elif family == 'l_plate':
        poly = Polygon([[0,0],[.052,0],[.052,.014],[.014,.014],[.014,.046],[0,.046]])
        mesh = trimesh.creation.extrude_polygon(poly, .010, engine='earcut')
        mesh.apply_scale(s)
    else:
        poly = Polygon([[0,0],[.050,0],[.050,.045],[.038,.045],[.038,.012],[.012,.012],[.012,.045],[0,.045]])
        mesh = trimesh.creation.extrude_polygon(poly, .020, engine='earcut')
        mesh.apply_scale(s)
    bounds = mesh.bounds
    mesh.apply_translation([-bounds[:,0].mean(), -bounds[:,1].mean(), -bounds[0,2]])
    if mesh.volume < 0:
        mesh.invert()
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    return mesh


def camera_pose(position, target):
    z = np.asarray(target)-position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.,0.,1.]); x /= np.linalg.norm(x)
    y = np.cross(z,x)
    T = np.eye(4); T[:3,:3] = np.column_stack([x,y,z]); T[:3,3] = position
    return T


def cameras():
    middle = np.array([0.,-.34,.47])
    central = camera_pose(middle, np.array([0.,0.,.015]))
    result = {}
    poses = {}
    for name, sign in [('left',-1),('right',1)]:
        T = central.copy(); T[:3,3] += sign*BASELINE/2*T[:3,0]; poses[name] = T
    poses['aux_left'] = camera_pose(np.array([-.36,.20,.42]), np.array([0.,0.,.015]))
    poses['aux_right'] = camera_pose(np.array([.36,.20,.42]), np.array([0.,0.,.015]))
    for name,T in poses.items():
        result[name] = dict(image=name+'.png', K=[[FOCAL,0,WIDTH/2],[0,FOCAL,HEIGHT/2],[0,0,1]],
                            world_from_camera=T.tolist(), width=WIDTH, height=HEIGHT,
                            distortion=[0,0,0,0,0])
    return dict(schema_version=1, frame_id='dataset_world', units='m',
                optical_convention='opencv_x_right_y_down_z_forward', world_up='+z',
                stereo_pair=['left','right'], stereo_baseline_m=BASELINE,
                multiview_names=list(result), cameras=result,
                calibration_source='exact synthetic pinhole poses; no real calibration error model')


def pose(x,y,yaw):
    c,s=np.cos(yaw),np.sin(yaw)
    T=np.eye(4);T[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];T[:3,3]=[x,y,0]
    return T


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--poses',type=int,default=3);args=parser.parse_args()
    root=args.out.resolve()
    if root.exists():raise FileExistsError('Use a new immutable dataset directory: '+str(root))
    root.mkdir(parents=True)
    start=time.perf_counter();rng=np.random.default_rng(SEED)
    cams=cameras();write(root/'calibration/cameras.json',cams)
    meshes={};objects=[]
    for fidx,family in enumerate(FAMILIES):
        for variant in range(4):
            oid=f'{family}_{variant+1:02d}';mesh=make_part(family,variant);meshes[oid]=mesh
            folder=root/'assets'/oid;folder.mkdir(parents=True)
            mesh.export(folder/'mesh_m.ply')
            mm=mesh.copy();mm.apply_scale(1000);mm.export(folder/'robotstudio_mm.stl')
            objects.append(dict(object_id=oid,family=family,name_zh=NAMES[fidx],variant=variant+1,
                split='development' if family in DEV_FAMILIES else 'test', units='m',
                mesh='assets/'+oid+'/mesh_m.ply',robotstudio_mesh='assets/'+oid+'/robotstudio_mm.stl',
                robotstudio_mesh_units='mm',extents_m=mesh.extents.tolist(),volume_m3=float(mesh.volume),
                center_of_mass_local_m=mesh.center_mass.tolist(),watertight=True,
                source='original procedural industrial-shape proxy; not a manufacturer-certified part',
                manufacturing_details='threads, chamfers and tolerances omitted',
                mass_kg=float(mesh.volume*2700),density_kg_m3=2700,
                mass_source='assumed homogeneous aluminium proxy; not measured',
                grasp_success=None,abb_execution_status='not_run'))
    write(root/'objects.json',objects)
    renderer=pyrender.OffscreenRenderer(WIDTH,HEIGHT)
    episodes=[];previews=[]
    for idx,obj in enumerate(objects):
        oid=obj['object_id'];mesh=meshes[oid]
        for rep in range(args.poses):
            # Sample placement once, reuse across all three environmental conditions.
            T=pose(rng.uniform(-.035,.035),rng.uniform(-.025,.025),rng.uniform(-np.pi,np.pi))
            neighbours=[]
            pool=[o for o in objects if o['split']==obj['split'] and o['object_id']!=oid]
            for j,(x,y) in enumerate([[-.12,-.045],[.12,.045],[0,.13]]):
                other=pool[(idx+rep+j)%len(pool)]
                neighbours.append((other['object_id'],pose(x,y,rng.uniform(-np.pi,np.pi))))
            for condition in CONDITIONS:
                eid=f'{oid}_p{rep+1:02d}_{condition}';inp=root/'inputs'/eid;gt=root/'evaluation'/eid
                inp.mkdir(parents=True);gt.mkdir(parents=True)
                scene=pyrender.Scene(bg_color=[.72,.76,.80,1],ambient_light=[.34]*3)
                matte=pyrender.MetallicRoughnessMaterial(baseColorFactor=[.55,.58,.61,1],metallicFactor=.15,roughnessFactor=.75)
                target_node=scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False,material=matte),pose=T)
                seg={target_node:np.array([1,0,0])};instances=[dict(instance_id=1,object_id=oid,role='target',world_from_object=T.tolist())]
                table=trimesh.creation.box(extents=[.65,.5,.035]);table.apply_translation([0,0,-.0175])
                table.visual.vertex_colors=[72,90,102,255]
                node=scene.add(pyrender.Mesh.from_trimesh(table));seg[node]=[250,0,0]
                scene_assets=[dict(kind='box',name='table',extents_m=[.65,.5,.035],center_m=[0,0,-.0175])]
                if condition=='neighbours':
                    for iid,(other,To) in enumerate(neighbours,2):
                        node=scene.add(pyrender.Mesh.from_trimesh(meshes[other],smooth=False,material=matte),pose=To)
                        seg[node]=[iid,0,0];instances.append(dict(instance_id=iid,object_id=other,role='distractor',world_from_object=To.tolist()))
                if condition=='bin_walls':
                    for j,(ext,center) in enumerate([([.32,.012,.07],[0,-.105,.035]),([.32,.012,.07],[0,.105,.035]),([.012,.21,.07],[-.154,0,.035]),([.012,.21,.07],[.154,0,.035])]):
                        wall=trimesh.creation.box(extents=ext);wall.apply_translation(center);wall.visual.vertex_colors=[108,114,125,255]
                        node=scene.add(pyrender.Mesh.from_trimesh(wall));seg[node]=[251,0,0]
                        scene_assets.append(dict(kind='box',name=f'bin_wall_{j}',extents_m=ext,center_m=center))
                for pos,intensity in [(np.array([.2,-.2,.8]),2.4),(np.array([-.4,.3,.6]),1.2)]:
                    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=intensity),pose=camera_pose(pos,np.zeros(3))@np.diag([1,-1,-1,1]))
                truth={};visibility={}
                for name,c in cams['cameras'].items():
                    cp=np.array(c['world_from_camera'])@np.diag([1,-1,-1,1])
                    node=scene.add(pyrender.IntrinsicsCamera(FOCAL,FOCAL,WIDTH/2,HEIGHT/2,znear=.02,zfar=3),pose=cp)
                    scene.main_camera_node=node
                    rgb,depth=renderer.render(scene)
                    labels,_=renderer.render(scene,flags=pyrender.RenderFlags.SEG,seg_node_map=seg)
                    mask=labels[:,:,0].astype(np.uint8)
                    Image.fromarray(rgb).save(inp/(name+'.png'))
                    truth[name+'_depth_m']=depth.astype(np.float32);truth[name+'_instance_id']=mask
                    visibility[name]=int((mask==1).sum())
                    scene.remove_node(node)
                assert visibility['left']>64 and visibility['right']>64, (eid,visibility)
                write(inp/'cameras.json',cams)
                write(inp/'task.json',dict(episode_id=eid,split=obj['split'],task='pick_and_lift',
                    target_selection='workpiece in fixed central workspace ROI; use RGB/perceived points, not ground-truth masks',
                    target_roi_world_m={'min':[-.085,-.075,0],'max':[.085,.075,.12]},
                    purpose='perception_inference',geometry_source='rgb_images',
                    renderer_depth_provided=False,object_pose_provided=False))
                np.savez_compressed(gt/'truth.npz',**truth)
                write(gt/'scene.json',dict(episode_id=eid,units='m',frame_id='dataset_world',instances=instances,
                    scene_assets=scene_assets,condition=condition,matched_pose_id=f'{oid}_p{rep+1:02d}',
                    placement='analytic stable flat support with random planar translation and yaw; no dynamics settling',
                    visible_target_pixels=visibility,purpose='evaluation_and_simulator_only',
                    robot_world_from_dataset_world=None,abb_execution_status='not_run',grasp_success=None))
                row=dict(episode_id=eid,object_id=oid,family=obj['family'],split=obj['split'],condition=condition,
                    matched_pose_id=f'{oid}_p{rep+1:02d}',input_dir='inputs/'+eid,evaluation_dir='evaluation/'+eid,
                    camera_count=4,abb_execution_status='not_run',grasp_success=None)
                episodes.append(row)
                if rep==0 and condition=='single':previews.append((oid,inp/'left.png'))
                if len(episodes)%12==0:print(json.dumps({'rendered_episodes':len(episodes),'last':eid}),flush=True)
    renderer.delete()
    write(root/'episodes.json',episodes)
    for split in ['development','test']:
        write(root/'splits'/(split+'.json'),{'object_ids':[o['object_id'] for o in objects if o['split']==split],
            'families':sorted({o['family'] for o in objects if o['split']==split}),
            'episode_ids':[e['episode_id'] for e in episodes if e['split']==split]})
    sheet=Image.new('RGB',(4*320,6*260),'#edf1f4');draw=ImageDraw.Draw(sheet)
    for i,(oid,p) in enumerate(previews):
        # Crop actual rendered RGB for this preview only; input images remain untouched.
        view=Image.open(p).crop((160,120,480,360)).resize((320,240))
        x,y=(i%4)*320,(i//4)*260;sheet.paste(view,(x,y));draw.text((x+6,y+242),oid,fill='#102c3f')
    (root/'preview').mkdir();sheet.save(root/'preview/workpieces.jpg',quality=92)
    sample=root/'inputs'/episodes[0]['episode_id'];quad=Image.new('RGB',(1280,960))
    for i,name in enumerate(cams['cameras']):quad.paste(Image.open(sample/(name+'.png')),((i%2)*640,(i//2)*480))
    quad.save(root/'preview/four_cameras.jpg',quality=92)
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()}
    write(root/'manifest.json',dict(dataset='Industrial Workpieces v1',schema_version=1,status='rendered_awaiting_ABB_execution',
        object_count=len(objects),episode_count=len(episodes),rgb_image_count=len(episodes)*4,
        families=FAMILIES,conditions=CONDITIONS,split_counts={s:sum(e['split']==s for e in episodes) for s in ['development','test']},
        seed=SEED,poses_per_object_per_condition=args.poses,image_size=[WIDTH,HEIGHT],
        input_routes={'stereo':['left','right'],'multiview':list(cams['cameras'])},
        renderer='pyrender EGL on Huashuo; perspective RGB rasterization, not RobotStudio screenshots',
        geometry_and_images_generated=True,abb_execution_count=0,grasp_success_rate=None,
        rendering_seconds=time.perf_counter()-start,
        renderer_limitations=['procedural geometry proxies, no threads or manufacturing tolerances',
            'matte rasterized appearance; realistic metal reflections and sensor noise not validated',
            'no gravity/contact simulation and no robot reachability verification',
            'static single-target scenes; no random heaps or arbitrary roll/pitch'],
        environment={'python':platform.python_version(),**{m:importlib.metadata.version(m) for m in ['numpy','pyrender','trimesh','Pillow','shapely']}},
        generator_sha256=sha(Path(__file__)),files=files,total_bytes=sum(f['bytes'] for f in files.values())))
    print(json.dumps({'completed':True,'objects':len(objects),'episodes':len(episodes),'rgb_images':len(episodes)*4}),flush=True)


if __name__=='__main__':main()
