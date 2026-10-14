"""Render two deterministic examples from recorded PhysX poses, not animation targets."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import hashlib, json, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh, pyrender, imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont

RUN=Path(__file__).resolve().parents[1]; STUDY=RUN.parent
OUT=RUN/'visualization'

def matrix(p):
    H=np.eye(4);H[:3,3]=p[:3]
    H[:3,:3]=Rotation.from_quat(np.r_[p[4:],p[3]]).as_matrix();return H

def look(eye,target):
    eye=np.array(eye,dtype=float);z=eye-np.array(target);z/=np.linalg.norm(z)
    x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
    H=np.eye(4);H[:3,:3]=np.column_stack([x,np.cross(z,x),z]);H[:3,3]=eye;return H

def main():
    OUT.mkdir(exist_ok=True)
    records=json.loads((RUN/'inputs/manifest.json').read_text())['records']
    selected=[]
    for success in [True,False]:
        for rec in records:
            if rec['condition']!='single':continue
            result=json.loads((RUN/'results/arm/nominal'/rec['object_id']/'result.json').read_text())
            row=next(r for r in result['rows'] if r['case_id']==rec['case_id'])
            if row['execution_success']==success:
                selected.append((rec,row));break
    assert len(selected)==2
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    renderer=pyrender.OffscreenRenderer(800,580)
    metadata=[]
    writer=imageio.get_writer(OUT/'recorded_physics_replay.mp4',fps=10,codec='libx264',quality=8,macro_block_size=2,output_params=['-movflags','+faststart','-pix_fmt','yuv420p'])
    for rec,row in selected:
        tracepath=RUN/'results/arm/nominal'/rec['object_id']/'trace.npz'
        trace=np.load(tracepath); i=list(trace['case_ids']).index(rec['case_id'])
        origin=trace['origins'][i]; body_names=list(trace['body_names'])
        scene=pyrender.Scene(bg_color=[.88,.91,.94,1],ambient_light=[.35]*3)
        nodes=[]
        for link in ET.parse(RUN/'assets/fanuc_panda.urdf').findall('link'):
            if link.get('name') not in body_names:continue
            for visual in link.findall('visual'):
                entry=visual.find('geometry/mesh')
                if entry is None:continue
                mesh=trimesh.load(entry.get('filename'),force='mesh')
                mesh.apply_scale(np.fromstring(entry.get('scale','1 1 1'),sep=' '))
                o=visual.find('origin'); H=np.eye(4)
                if o is not None:
                    H[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                    H[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
                mesh.apply_transform(H)
                mesh.visual.vertex_colors=[226,178,48,255] if link.get('name').startswith('J') else [65,75,88,255]
                nodes.append((scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False)),body_names.index(link.get('name'))))
        cfg=json.loads((STUDY/rec['scene_source']).read_text())
        assert len(cfg['instances'])==1
        for item in cfg['scene_assets']:
            mesh=trimesh.creation.box(item['extents_m']);mesh.apply_translation(np.array(item['center_m'])+[.965,0,1.240])
            mesh.visual.vertex_colors=[135,151,161,255]
            scene.add(pyrender.Mesh.from_trimesh(mesh))
        objmesh=trimesh.load(STUDY/rec['mesh_source'],force='mesh');objmesh.visual.vertex_colors=[32,135,188,255]
        obj=scene.add(pyrender.Mesh.from_trimesh(objmesh,smooth=False))
        camera=scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(44)),pose=look([2.5,-2.6,2.5],[.5,0,.95]))
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2),pose=look([2,-2,3],[.8,0,1]))
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=.8),pose=look([-.5,1,2],[.8,0,1]))
        selected_frames={str(phase):int(np.flatnonzero(trace['phases']==phase)[len(np.flatnonzero(trace['phases']==phase))//2]) for phase in ['approach','close','hold','placed_hold']}
        snapshots=[]
        for t in sorted(set(range(0,len(trace['time_s']),5))|set(selected_frames.values())):
            for node,k in nodes:
                p=trace['bodies'][t,i,k].copy();p[:3]-=origin;scene.set_pose(node,matrix(p))
            p=trace['object_pose'][t,i].copy();p[:3]-=origin;scene.set_pose(obj,matrix(p))
            renderer.viewport_width=800;renderer.viewport_height=580
            scene.set_pose(camera,look([2.5,-2.6,2.5],[.5,0,.95]));rgb,_=renderer.render(scene)
            renderer.viewport_width=380;renderer.viewport_height=330
            center=np.array([.965,0,1.34]);scene.set_pose(camera,look(center+[.34,-.36,.22],center));close,_=renderer.render(scene)
            frame=Image.new('RGB',(1200,720),'#102030');frame.paste(Image.fromarray(rgb),(12,90));frame.paste(Image.fromarray(close),(814,90))
            d=ImageDraw.Draw(frame)
            d.text((18,15),'Recorded Isaac Sim / PhysX | FANUC M-20iD/25 + Panda',font=font,fill='white')
            d.text((18,48),f"{rec['case_id']} | {rec['perception']} + {rec['grasp']} | {rec['episode_id']}",font=small,fill='#b8d2e8')
            lines=[f"Phase: {trace['phases'][t]}",f"Simulation time: {trace['time_s'][t]:.2f} s",f"Lift + hold: {row['physics_lift_success']}",f"Endpoint reached: {row['approach_endpoint_reached']}",f"No prohibited contact: {row['collision_free']}",f"Execution pass: {row['execution_success']}",f"Object attachment: False"]
            for j,line in enumerate(lines):d.text((827,445+29*j),line,font=small,fill='white')
            d.text((18,687),'Frozen candidate and joint path; images render recorded body/object poses. No new perception inference.',font=small,fill='#b8d2e8')
            if t in selected_frames.values():
                file=OUT/f"{rec['case_id']}_{trace['phases'][t]}.png";frame.save(file);snapshots.append(str(file.relative_to(RUN)))
            if t%5==0:writer.append_data(np.asarray(frame))
        metadata.append(dict(case_id=rec['case_id'],selection='First single-object record by case_id with this execution outcome',
            outcome=row,trace_sha256=hashlib.sha256(tracepath.read_bytes()).hexdigest(),snapshots=snapshots))
        print('Rendered',rec['case_id'],row['execution_success'],flush=True)
    writer.close();renderer.delete()
    (OUT/'video_manifest.json').write_text(json.dumps(dict(fps=10,simulation_seconds_per_case=19.5,cases=metadata,
        rendering='Recorded PhysX robot-link and target poses; no pose interpolation or synthetic outcome'),indent=2)+'\n')

if __name__=='__main__':main()
