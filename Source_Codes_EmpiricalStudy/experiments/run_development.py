#!/usr/bin/env python3
"""Auditable development-only launch, preserving failures and all raw artifacts."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
GRASP=PROJECT/'Source_Codes_Grasping_Modules'
DATA=HERE.parent/'datasets/industrial_workpieces_v1'
PY=GRASP/'baseline_graspgen/.venv-infer/bin/python'
STEREO=['stereobm','sgm','foundationstereo','psmnet','raft_stereo','igev_stereo','gwcnet']
GRASPS=['graspgen','6dof_graspnet','contact_graspnet','gpd','graspnet_baseline']


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require_launch_enabled():
    control=json.loads((HERE/'task_control.json').read_text())
    if not control.get('launch_enabled',False):
        raise SystemExit('Experiment stopped by user. FANUC is selected; a new experiment has not been started.')
def write(p,d):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');temp.replace(p)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--all-development',action='store_true')
    p.add_argument('--perception-only',action='store_true');a=p.parse_args()
    require_launch_enabled()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'logs').mkdir()
    episodes=[f'{obj}_p01_{condition}' for obj in ['spacer_01','mounting_block_01']
              for condition in ['single','neighbours','bin_walls']]
    if a.all_development:
        episodes=[e['episode_id'] for e in json.loads((DATA/'episodes.json').read_text()) if e['split']=='development']
        assert len(episodes)==72
    snapshot=a.out/'source_snapshot';snapshot.mkdir()
    for path in HERE.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    manifest=dict(run_id=a.out.name,started_at=now(),pid=os.getpid(),split='development',episodes=episodes,
        research_questions=['RQ1 point-cloud quality','RQ2 fixed GraspGen','RQ3 fixed FoundationStereo'],
        phase='development perception' if a.perception_only else 'development perception and candidate generation',test_set_started=False,
        execution_platform='Perception and candidate generation only',native_fanuc_executions=0,fanuc_grasp_success_rate=None,
        method_selection='7 calibrated RGB stereo methods; multiview excluded by user on 2026-10-12',
        input_camera_names=['left','right'],
        method_parameters_frozen_for_this_run=True,seed=0,
        dataset_manifest_sha256=sha(DATA/'manifest.json'),
        script_sha256={x.name:sha(x) for x in HERE.glob('*.py')},
        perception_methods=STEREO,grasp_methods=GRASPS,
        grasps_for_RQ2='graspgen',perceptions_for_RQ3=['foundationstereo'],
        status='running',events=[])
    write(a.out/'status.json',manifest)
    locks=[]
    for name in ['stereo','grasp']:
        lock=open('/tmp/robotic_grasping_'+name+'_eval.lock','a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    def run(label,cmd,timeout):
        event=dict(label=label,status='running',started_at=now(),command=list(map(str,cmd)))
        manifest['events'].append(event);write(a.out/'status.json',manifest)
        start=time.perf_counter()
        with (a.out/'logs'/(label+'.log')).open('x') as log:
            child=subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=child.wait(timeout=timeout);status='completed' if rc==0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                rc=child.returncode;status='timeout'
        event.update(status=status,returncode=rc,finished_at=now(),wall_seconds=time.perf_counter()-start,
                     log_sha256=sha(a.out/'logs'/(label+'.log')))
        write(a.out/'status.json',manifest);print(json.dumps(event),flush=True)
        return rc==0
    for method in STEREO:
        cmd=[PY,HERE/'perception.py','--method',method]
        cmd += ['--inputs',DATA/'inputs','--episodes',*episodes,'--out',a.out/'perception'/method]
        run('perception_'+method,cmd,1800)
        for episode in episodes:
            prediction=a.out/'perception'/method/episode
            if (prediction/'cloud.npz').exists():
                run('geometry_'+method+'_'+episode,[PY,HERE/'evaluate_perception.py','--dataset',DATA,
                    '--episode',episode,'--prediction',prediction],120)
    for method in ([] if a.perception_only else STEREO):
        backends=GRASPS if method=='foundationstereo' else ['graspgen']
        for episode in episodes:
            source=a.out/'perception'/method/episode/'grasp_input.npz'
            for backend in backends:
                label='grasp_'+method+'_'+episode+'_'+backend
                if not source.exists():
                    manifest['events'].append(dict(label=label,status='no_perception_input',
                        algorithm_failure=True,native_robot_execution_performed=False))
                    write(a.out/'status.json',manifest);continue
                run(label,[PY,GRASP/'shared/infer.py','--method',backend,'--input',source,
                    '--out',a.out/'grasps'/method/episode/backend],900)
    manifest.update(status='development_inference_finished_fanuc_pending',finished_at=now())
    write(a.out/'status.json',manifest)


if __name__=='__main__':main()
