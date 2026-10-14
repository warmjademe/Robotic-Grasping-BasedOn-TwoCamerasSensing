"""Official PhysX tug evaluation with explicit drives and saved closure commands.

Upstream is left untouched. Adapters set mass/material/actuator parameters and
replace only the all-zero closure command with the frozen plan's command.
"""
import argparse, fcntl, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('GRASPGEN_RUNTIME','/home/qyb/RESEARCH/baseline_graspgen'))
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
sys.path.insert(0,str(BASE/'upstream/GraspDataGen/scripts/graspgen'))
os.chdir(BASE/'upstream/GraspDataGen')
import grasp_sim
from graspgen_utils import start_isaac_lab_if_needed

def main():
    parser=argparse.ArgumentParser()
    grasp_sim.add_grasp_sim_args(parser,{},**grasp_sim.collect_grasp_sim_args({}))
    parser.add_argument('--output',required=True)
    parser.add_argument('--friction',type=float,default=1.)
    args=parser.parse_args(); args.headless=True
    import yaml
    source=Path(args.grasp_file); data=yaml.safe_load(source.read_text())
    args.object_file=data['object_file']; args.object_scale=1.
    args.obj2usd_friction=args.friction
    args.obj2usd_collision_approximation='convexDecomposition'
    args.obj2usd_use_existing_usd=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists(): raise RuntimeError('Refusing to overwrite result')
    lock=(RUN/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    app=start_isaac_lab_if_needed(__file__,headless=True)
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene
    trace=[]; actual={}; state={'steps':0,'start_idx':0}
    orig_update=InteractiveScene.update; orig_target=Articulation.set_joint_position_target
    simulator=grasp_sim.GraspingSimulation.from_args(args)
    entries=list(data['grasps'].items())
    old_cfg=simulator.build_grasp_sim_scene_cfg
    def configure(n):
        cfg=old_cfg(n)
        a=cfg.gripper.actuators['gripper'];a.stiffness=1000.;a.damping=100.;a.effort_limit_sim=20.
        cfg.object.spawn.mass_props=sim_utils.MassPropertiesCfg(mass=.1)
        cfg.object.spawn.collision_props.contact_offset=.0005
        cfg.object.spawn.collision_props.rest_offset=0.
        return cfg
    simulator.build_grasp_sim_scene_cfg=configure
    old_init=simulator.get_initial_joint_pos
    def initialize(scene,joint_pos,n,start_idx,buff):
        state['start_idx']=start_idx
        return old_init(scene,joint_pos,n,start_idx,buff)
    simulator.get_initial_joint_pos=initialize
    def target(robot,target,*a,**kw):
        # Isaac's soft lower limit is slightly above zero (joint-limit margin).
        # Match the exact upstream closure expression, not a numeric zero.
        if 'Robot' in robot.cfg.prim_path and torch.allclose(target,robot.data.soft_joint_pos_limits[...,0]):
            values=[entries[int(simulator.grasp_idx_map[i])][1]['closing_command_m']/2
                    for i in range(state['start_idx'],state['start_idx']+target.shape[0])]
            target=torch.tensor(values,device=target.device,dtype=target.dtype)[:,None].expand_as(target).clone()
            state['closure_commands']=target.detach().cpu().tolist()
        return orig_target(robot,target,*a,**kw)
    Articulation.set_joint_position_target=target
    def update(scene,dt):
        orig_update(scene,dt);state['steps']+=1
        if not actual:
            actual.update(mass_kg=scene['object'].data.default_mass[0].cpu().tolist(),
                joints=scene['gripper'].joint_names,
                stiffness=scene['gripper'].data.joint_stiffness[0].cpu().tolist(),
                damping=scene['gripper'].data.joint_damping[0].cpu().tolist())
        if state['steps']%5==0:
            trace.append(dict(object_pose=scene['object'].data.root_state_w[:,:7].cpu().numpy().copy(),
                joints=scene['gripper'].data.joint_pos.cpu().numpy().copy(),
                finger0_forces=scene['contact_forces0'].data.net_forces_w.cpu().numpy().copy(),
                finger1_forces=scene['contact_forces1'].data.net_forces_w.cpu().numpy().copy(),
                origins=scene.env_origins.cpu().numpy().copy()))
    InteractiveScene.update=update
    start=time.perf_counter()
    buff=simulator.validate_grasps()
    assert buff is not None
    labels=buff.is_success.numpy().astype(bool)
    rows=[dict(case_id=entries[int(simulator.grasp_idx_map[i])][0],success=bool(v),
               closing_command_m=entries[int(simulator.grasp_idx_map[i])][1]['closing_command_m'],
               expected=entries[int(simulator.grasp_idx_map[i])][1].get('expected')) for i,v in enumerate(labels)]
    report=dict(stage='isolated_gripper_tug',status='completed',rows=rows,actual_parameters=actual,
       seconds=time.perf_counter()-start,friction=args.friction,fps=args.fps,
       gravity_enabled=False,force_magnitude_g=args.force_magnitude,applied_closure_commands=state.get('closure_commands'),
       tug_sequences=simulator.tug_sequences,closure='saved_plan_closing_command',
       object_attached=False,full_arm=False,
       success_definition='Official GraspDataGen: both fingers have nonzero contact force after final tug',
       source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
       code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       upstream_sha256=hashlib.sha256(Path(grasp_sim.__file__).read_bytes()).hexdigest(),
       versions={n:importlib.metadata.version(n) for n in ['isaacsim','isaaclab','torch','warp-lang']})
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'trace.npz',**{k:np.stack([t[k] for t in trace]) for k in trace[0]})
    simulator.create_isaac_grasp_data(buff,save_successes=True,save_fails=True,save_to_folder=str(out/'official'))
    print('RESULT',json.dumps(dict(rows=rows,seconds=report['seconds'])),flush=True)
    app.close(wait_for_replicator=False,skip_cleanup=True)

if __name__=='__main__':main()
