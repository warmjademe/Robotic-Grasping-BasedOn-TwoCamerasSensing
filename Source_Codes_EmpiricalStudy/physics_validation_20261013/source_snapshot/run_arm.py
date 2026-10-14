"""Replay frozen FANUC paths in Isaac/PhysX with physical Panda fingers.

No object attachment or object pose writes after initialization. CAD belongs
only to the evaluation scene. Captures full trajectories and contact forces.
"""
import argparse, fcntl, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];STUDY=RUN.parent
BASE=Path('/home/qyb/RESEARCH/baseline_graspgen')
parser=argparse.ArgumentParser()
parser.add_argument('--group',default='calibration')
parser.add_argument('--output',required=True)
parser.add_argument('--fps',type=int,default=250)
parser.add_argument('--friction',type=float,default=1.)
args=parser.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():raise RuntimeError('Refusing to overwrite completed run')
lock=(RUN/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True).app
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import isaaclab.sim as su
from isaaclab.assets import Articulation,ArticulationCfg,RigidObject,RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
from isaaclab.sim.converters import UrdfConverter,UrdfConverterCfg
from isaaclab.sim.converters.asset_converter_base import AssetConverterBase
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Sdf
import omni.usd
sys.path.insert(0,str(BASE/'upstream/GraspDataGen/scripts/graspgen'))
from mesh_utils import convert_mesh_to_usd
sys.path.insert(0,str(STUDY.parent/'Source_Codes_Grasping_Modules/baseline_graspgen/fanuc'))
from kinematics import fk,solve,translation,from_wpr

class InstalledImporter(UrdfConverter):
    def __init__(self,cfg):
        import omni.kit.app
        manager=omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('isaacsim.asset.importer.urdf',True)
        from isaacsim.asset.importer.urdf._urdf import acquire_urdf_interface
        self._urdf_interface=acquire_urdf_interface()
        AssetConverterBase.__init__(self,cfg=cfg)
    def _get_urdf_import_config(self):
        import omni.kit.commands
        _,cfg=omni.kit.commands.execute('URDFCreateImportConfig')
        for key,value in dict(distance_scale=1.,make_default_prim=True,create_physics_scene=False,
            density=1000.,convex_decomp=True,collision_from_visuals=False,merge_fixed_joints=False,
            fix_base=True,self_collision=True,parse_mimic=False,replace_cylinders_with_capsules=False).items():
            getattr(cfg,'set_'+key)(value)
        return cfg

def serial(q):
    q=np.array(q,dtype=float).copy();q[...,2]+=q[...,1];return q
def pose7(H):
    q=Rotation.from_matrix(H[:3,:3]).as_quat();return np.r_[H[:3,3],q[3],q[:3]]
def tensor(a):return torch.as_tensor(np.asarray(a),dtype=torch.float32,device='cuda:0')
def interpolate(path,f):
    if len(path)==1:return path[0]
    x=min(max(f,0.),1.)*(len(path)-1);j=min(int(x),len(path)-2)
    return path[j]*(1-(x-j))+path[j+1]*(x-j)

manifest=json.loads((RUN/'inputs/manifest.json').read_text())
records=[r for r in manifest['records'] if r['object_id']==args.group]
scenes=[];selections=[];meshfiles=[]
if args.group=='calibration':
    H=np.eye(4);H[:3,:3]=np.diag([1,-1,-1]);H[:3,3]=[.965,0,1.3565]
    calibration_mesh=RUN/'assets/arm_calibration_cube.obj'
    trimesh.creation.box([.03,.03,.03]).export(calibration_mesh)
    F=H@translation([0,0,-.04]);pre=F.copy();pre[2,3]+=.12
    lift=F.copy();lift[2,3]+=.15
    home=from_wpr([800,160,1540,180,0,0]);qhome,err=solve(home,np.zeros(6));assert max(err)<.01
    paths={};seed=qhome
    qpre,err=solve(pre,seed);assert max(err)<.01
    paths['transit']=np.linspace(qhome,qpre,41)
    for phase,A,B in [('approach',pre,F),('lift',F,lift)]:
        qs=[];seed=qpre if phase=='approach' else qs_grasp
        for f in np.linspace(0,1,31):
            T=A.copy();T[:3,3]=A[:3,3]*(1-f)+B[:3,3]*f
            seed,err=solve(T,seed);assert max(err)<.01;qs.append(seed.copy())
        paths[phase]=np.array(qs);qs_grasp=qs[-1].copy()
    records=[]
    for name,dx,op in [('centered_box',0.,False),('far_negative',.2,False),('open_negative',0.,True),('collision_blocked',.2,False)]:
        records.append(dict(case_id=name,expected=name=='centered_box',jaw_opening_m=.08,closing_command_m=.08 if op else 0.))
        scenes.append(dict(instances=[dict(object_id='calibration_box',role='target',world_from_object=translation([dx,0,.015]).tolist())],
            scene_assets=[dict(name='table',kind='box',extents_m=[.65,.5,.035],center_m=[0,0,-.0175])]))
        if name=='collision_blocked':
            scenes[-1]['scene_assets'].append(dict(name='calibration_obstacle',kind='box',extents_m=[.15,.15,.10],center_m=[0,0,.08]))
        selections.append(dict(paths=paths,hand_pose=H,opening=.08,closing=.08 if op else 0.))
        meshfiles.append(calibration_mesh)
else:
    for r in records:
        plan=json.loads((STUDY/r['plan_source']).read_text())['selected']
        scenes.append(json.loads((STUDY/r['scene_source']).read_text()))
        paths={phase:np.deg2rad([x['joints_deg'] for x in plan['path'] if x['phase']==phase]) for phase in ['transit','approach','lift']}
        selections.append(dict(paths=paths,hand_pose=translation([.965,0,1.240])@np.array(plan['hand_pose_dataset_m']),
            opening=plan['jaw_opening_m'],closing=plan['closing_command_m']))
        meshfiles.append(STUDY/r['mesh_source'])
assert records
n=len(records); origins=np.array([[i%6*5.,i//6*5.,0.] for i in range(n)])
started=time.perf_counter()
converter=InstalledImporter(UrdfConverterCfg(asset_path=str(RUN/'assets/fanuc_panda.urdf'),
    usd_dir=str(RUN/'assets/full_fanuc'),usd_file_name='fanuc.usd',fix_base=True,
    merge_fixed_joints=False,make_instanceable=False,convert_mimic_joints_to_normal_joints=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1.e6,damping=1.e5))))
sim=su.SimulationContext(su.SimulationCfg(dt=1/args.fps,device='cuda:0',
    physx=su.PhysxCfg(min_position_iteration_count=32,min_velocity_iteration_count=4)))
stage=omni.usd.get_context().get_stage()
for i in range(n):UsdGeom.Xform.Define(stage,f'/World/envs/env_{i}')
material=su.RigidBodyMaterialCfg(static_friction=args.friction,dynamic_friction=args.friction,restitution=0.)
cfg=ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
    spawn=su.UsdFileCfg(usd_path=converter.usd_path,activate_contact_sensors=True,
        rigid_props=su.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=su.ArticulationRootPropertiesCfg(enabled_self_collisions=True,solver_position_iteration_count=32,solver_velocity_iteration_count=4)),
    actuators={'arm':ImplicitActuatorCfg(joint_names_expr=['J[1-6]'],stiffness=1.e6,damping=1.e5),
       'gripper':ImplicitActuatorCfg(joint_names_expr=['panda_finger_joint.*'],stiffness=1000.,damping=100.,effort_limit_sim=20.)})
robot=Articulation(cfg)
# Match documented adjacent housing overlap exception in the original planner.
for i in range(n):
    base=f'/World/envs/env_{i}/Robot'
    for a,b in [('J4_link','J6_link'),('J6_link','panda_hand')]:
        pa=stage.GetPrimAtPath(base+'/'+a);pb=stage.GetPrimAtPath(base+'/'+b)
        if pa.IsValid() and pb.IsValid():UsdPhysics.FilteredPairsAPI.Apply(pa).CreateFilteredPairsRel().AddTarget(pb.GetPath())
# Set actual finger/object friction explicitly, including imported USD materials.
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MaterialAPI):
        mat=UsdPhysics.MaterialAPI(prim);mat.CreateStaticFrictionAttr(args.friction);mat.CreateDynamicFrictionAttr(args.friction)

usd_cache={}
def object_usd(mesh):
    key=hashlib.sha256(Path(mesh).read_bytes()).hexdigest()
    if key not in usd_cache:
        dest=RUN/'assets'/('object_'+key[:12]+f'_f{args.friction}.usd')
        convert_mesh_to_usd(str(dest),str(mesh),mass=.1,collision_approximation='convexDecomposition',physics_material=material)
        usd_cache[key]=str(dest)
    return usd_cache[key]

objects=[];meshes=[];distractors=[];table_z=1.240
for i,(r,scene,meshpath) in enumerate(zip(records,scenes,meshfiles)):
    prefix=f'/World/envs/env_{i}'
    UsdGeom.Xform.Define(stage,prefix+'/Fixtures')
    for fixture in scene['scene_assets']:
        name='Table' if fixture['name']=='table' else 'Wall_'+fixture['name']
        c=su.CuboidCfg(size=tuple(fixture['extents_m']),collision_props=su.CollisionPropertiesCfg(contact_offset=.0005,rest_offset=0.),physics_material=material)
        xyz=origins[i]+[.965,0,table_z]+np.array(fixture['center_m'])
        c.func(prefix+'/Fixtures/'+name,c,translation=tuple(xyz))
    for inst in scene['instances']:
        target=inst['role']=='target'
        mp=meshpath if target else STUDY/'datasets/industrial_workpieces_v1/assets'/inst['object_id']/'mesh_m.ply'
        T=translation(origins[i]+[.965,0,table_z])@np.array(inst['world_from_object'])
        ps=pose7(T)
        obj=RigidObject(RigidObjectCfg(prim_path=prefix+('/Target' if target else '/Distractor_'+str(inst['instance_id'])),
            spawn=su.UsdFileCfg(usd_path=object_usd(mp),activate_contact_sensors=True,
                rigid_props=su.RigidBodyPropertiesCfg(disable_gravity=False),
                collision_props=su.CollisionPropertiesCfg(contact_offset=.0005,rest_offset=0.),
                mass_props=su.MassPropertiesCfg(mass=.1)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(ps[:3]),rot=tuple(ps[3:]))))
        if target:objects.append(obj);meshes.append(trimesh.load(mp,force='mesh'))
        else:distractors.append(obj)
contact=ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*',update_period=0.,history_length=0))
finger_contacts=[ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/'+name,
    update_period=0.,history_length=0,filter_prim_paths_expr=['/World/envs/env_.*/Fixtures/.*','/World/envs/env_.*/Target']))
    for name in ['panda_leftfinger','panda_rightfinger']]
sim.reset()
robot.update(1/args.fps)
print('ROBOT',robot.joint_names,robot.body_names,flush=True)
arm_ids=[robot.joint_names.index('J'+str(i)) for i in range(1,7)]
finger_ids=[robot.joint_names.index('panda_finger_joint'+str(i)) for i in [1,2]]
hand_id=robot.body_names.index('panda_hand')
root=robot.data.default_root_state.clone();root[:,:3]=tensor(origins)
robot.write_root_pose_to_sim(root[:,:7]);robot.write_root_velocity_to_sim(root[:,7:])
q0=np.stack([serial(s['paths']['transit'][0]) for s in selections])
joints=robot.data.default_joint_pos.clone();joints[:,arm_ids]=tensor(q0)
joints[:,finger_ids]=tensor([[s['opening']/2]*2 for s in selections])
robot.write_joint_state_to_sim(joints,torch.zeros_like(joints));robot.set_joint_position_target(joints)
frames=[];dt=1/args.fps;elapsed=0.;step=0
bad_contact_steps=np.zeros(n,dtype=int);max_fixture_force=np.zeros(n);max_arm_force=np.zeros(n)
phases=[('settle',1.),('transit',3.),('approach',2.),('close',1.5),('lift',3.),('hold',2.),('lower',3.),('release',1.),('retreat',2.),('placed_hold',1.)]
phase_checks={};baseline=None
for phase,duration in phases:
    for j in range(round(duration*args.fps)):
        f=(j+1)/round(duration*args.fps);smooth=f*f*(3-2*f)
        goals=[]
        for s in selections:
            paths=s['paths']
            if phase=='settle':q=paths['transit'][0]
            elif phase in ['transit','approach','lift']:q=interpolate(paths[phase],smooth)
            elif phase=='close':q=paths['approach'][-1]
            elif phase=='hold':q=paths['lift'][-1]
            elif phase=='lower':q=interpolate(paths['lift'],1-smooth)
            elif phase=='release':q=paths['approach'][-1]
            elif phase=='retreat':q=interpolate(paths['approach'],1-smooth)
            else:q=paths['approach'][0]
            goals.append(serial(q))
        opening=np.array([s['opening'] if phase in ['settle','transit','approach','release','retreat','placed_hold'] else s['closing'] for s in selections])
        robot.set_joint_position_target(tensor(goals),joint_ids=arm_ids)
        robot.set_joint_position_target(tensor(np.repeat(opening[:,None]/2,2,axis=1)),joint_ids=finger_ids)
        robot.write_data_to_sim();sim.step(render=False);robot.update(dt);contact.update(dt)
        for sensor in finger_contacts:sensor.update(dt)
        for obj in objects+distractors:obj.update(dt)
        # Exclude expected finger contact with target; count housing contact and
        # finger contact with static fixtures separately at every physics step.
        net=contact.data.net_forces_w
        arm_contact_ids=[k for k,name in enumerate(contact.body_names) if name not in ['panda_leftfinger','panda_rightfinger','base_link']]
        arm_force=torch.linalg.vector_norm(net[:,arm_contact_ids,:],dim=-1).max(dim=-1).values.cpu().numpy()
        fm=np.stack([s.data.force_matrix_w.cpu().numpy().copy() for s in finger_contacts],axis=1)
        # Static colliders have no rigid-body root for the SDK's filtered view.
        # Subtract target-specific force from total finger force instead; this
        # also detects unintended contacts with neighbouring workpieces.
        finger_total=np.stack([s.data.net_forces_w.cpu().numpy().copy() for s in finger_contacts],axis=1)
        fixture_force=np.linalg.norm(finger_total-fm[:,:,:,-1,:],axis=-1).max(axis=(1,2))
        if phase not in ['settle']:
            bad_contact_steps+=(arm_force>.5)|(fixture_force>.5)
            max_fixture_force=np.maximum(max_fixture_force,fixture_force)
            max_arm_force=np.maximum(max_arm_force,arm_force)
        step+=1;elapsed+=dt
        if step%5==0:
            poses=np.stack([o.data.root_state_w[0,:7].cpu().numpy().copy() for o in objects])
            frames.append(dict(phase=phase,time=elapsed,object_pose=poses,
                joints=robot.data.joint_pos.cpu().numpy().copy(),bodies=robot.data.body_link_state_w[:,:,:7].cpu().numpy().copy(),
                contact=contact.data.net_forces_w.cpu().numpy().copy(),finger_contact_matrix=fm))
    if phase=='settle':baseline=np.stack([o.data.root_state_w[0,:7].cpu().numpy() for o in objects])
    if phase=='approach':
        actual=robot.data.body_link_state_w[:,hand_id,:7].cpu().numpy()
        expected=np.stack([pose7(translation(origins[i])@s['hand_pose']) for i,s in enumerate(selections)])
        pe=np.linalg.norm(actual[:,:3]-expected[:,:3],axis=1)*1000
        angle=[]
        for a,b in zip(actual,expected):angle.append(float((Rotation.from_quat(a[[4,5,6,3]]).inv()*Rotation.from_quat(b[[4,5,6,3]])).magnitude()*180/np.pi))
        phase_checks['approach_position_error_mm']=pe.tolist();phase_checks['approach_rotation_error_deg']=angle
    print('PHASE',phase,'target_z',[round(float(o.data.root_pos_w[0,2]),4) for o in objects],flush=True)

ph=np.array([x['phase'] for x in frames]);poses=np.stack([x['object_pose'] for x in frames])
body=np.stack([x['bodies'] for x in frames]);contact_forces=np.stack([x['contact'] for x in frames])
rows=[]
for i,r in enumerate(records):
    p=poses[:,i]; rot=Rotation.from_quat(p[:,[4,5,6,3]]).as_matrix()
    min_z=np.min(np.einsum('tv,pv->tp',rot[:,2,:],meshes[i].vertices),axis=1)+p[:,2]
    hold=min_z[ph=='hold']-table_z
    lift_pass=bool(hold.min()>=.1)
    ep=phase_checks['approach_position_error_mm'][i];er=phase_checks['approach_rotation_error_deg'][i]
    endpoint=ep<=2 and er<=.5
    placed=p[ph=='placed_hold'];placed_bottom=min_z[ph=='placed_hold']-table_z
    xyerror=float(np.linalg.norm(placed[-1,:2]-baseline[i,:2]))
    placement=bool(lift_pass and np.all(np.abs(placed_bottom)<.003) and xyerror<=.02)
    collision_free=bool(bad_contact_steps[i]==0)
    rows.append(dict(case_id=r['case_id'],expected=r.get('expected'),physics_lift_success=lift_pass,
        collision_free=collision_free,prohibited_contact_steps=int(bad_contact_steps[i]),
        max_finger_non_target_force_n=float(max_fixture_force[i]),max_arm_contact_force_n=float(max_arm_force[i]),
        execution_success=bool(lift_pass and endpoint and collision_free),
        approach_endpoint_reached=endpoint,approach_position_error_mm=ep,approach_rotation_error_deg=er,
        min_hold_bottom_clearance_m=float(hold.min()),hold_sim_s=2.,returned_to_origin_success=placement,
        return_xy_error_m=xyerror,final_bottom_clearance_m=float(placed_bottom[-1]),
        initial_settling_translation_m=float(np.linalg.norm(baseline[i,:3]-objects[i].data.default_root_state[0,:3].cpu().numpy())),
        closing_command_m=selections[i]['closing'],object_attached=False))
np.savez_compressed(out/'trace.npz',case_ids=np.array([r['case_id'] for r in records]),phases=ph,
    time_s=np.array([x['time'] for x in frames]),object_pose=poses,bodies=body,
    joints=np.stack([x['joints'] for x in frames]),contact_forces=contact_forces,
    body_names=np.array(robot.body_names),joint_names=np.array(robot.joint_names),
    contact_body_names=np.array(contact.body_names),origins=origins,
    finger_contact_matrix=np.stack([x['finger_contact_matrix'] for x in frames]))
report=dict(stage='full_arm_physics',status='completed',rows=rows,seconds=time.perf_counter()-started,
    fps=args.fps,friction=args.friction,object_mass_kg=.1,gravity_m_s2=9.81,
    model='FANUC official m20_25_18d URDF + Panda gripper',controller='Isaac PhysX implicit joint drives',
    object_attached=False,object_pose_written_after_initialization=False,phases=phases,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    manifest_sha256=hashlib.sha256((RUN/'inputs/manifest.json').read_bytes()).hexdigest(),
    versions={n:importlib.metadata.version(n) for n in ['isaacsim','isaaclab','torch']},
    collision_definition='Any nonfinger housing contact or finger non-target resultant normal force >0.5 N after settling',
    friction_scope='Explicit target/distractor/table material; robot uses imported USD/default material',
    notes='Lift, endpoint tracking, prohibited contact and placement are reported separately.')
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('RESULT',json.dumps(report),flush=True)
app.close(wait_for_replicator=False,skip_cleanup=True)
