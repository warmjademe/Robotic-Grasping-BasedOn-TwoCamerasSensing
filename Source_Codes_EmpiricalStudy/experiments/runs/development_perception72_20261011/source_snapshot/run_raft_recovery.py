#!/usr/bin/env python3
"""Retry the recorded missing-dependency failure, without changing predictions."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from run_development import HERE, DATA, PY, GRASP, now, sha, write

run=HERE/'runs/development_pilot_20261011'
out=HERE/'runs/development_raft_dependency_recovery_20261011'
out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir()
episodes=json.loads((run/'status.json').read_text())['episodes']
record=dict(status='waiting_for_gpu',started_at=now(),pid=os.getpid(),events=[],split='development',
    reason='First process failed on import: opt_einsum was absent; add isolated opt_einsum 3.4.0 dependency',
    original_failure_log_sha256=sha(run/'logs/perception_raft_stereo.log'),
    source_sha256={n:sha(HERE/n) for n in ['perception.py','run_raft_recovery.py']},
    original_failures_preserved=True,test_set_started=False)
write(out/'status.json',record)
locks=[]
for name in ['stereo','grasp']:
    handle=open('/tmp/robotic_grasping_'+name+'_eval.lock','a')
    fcntl.flock(handle,fcntl.LOCK_EX);locks.append(handle)
record['status']='running';write(out/'status.json',record)
commands=[('perception_raft_stereo',[PY,HERE/'perception.py','--method','raft_stereo','--inputs',DATA/'inputs',
    '--episodes',*episodes,'--out',out/'perception/raft_stereo'])]
for ep in episodes:
    prediction=out/'perception/raft_stereo'/ep
    commands.append(('geometry_raft_stereo_'+ep,[PY,HERE/'evaluate_perception.py','--dataset',DATA,'--episode',ep,'--prediction',prediction]))
    commands.append(('grasp_raft_stereo_'+ep,[PY,GRASP/'shared/infer.py','--method','graspgen',
        '--input',prediction/'grasp_input.npz','--out',out/'grasps/raft_stereo'/ep/'graspgen']))
for label,command in commands:
    event=dict(label=label,status='running',started_at=now());record['events'].append(event);write(out/'status.json',record)
    started=time.perf_counter()
    with (out/'logs'/(label+'.log')).open('x') as log:
        result=subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,timeout=900)
    event.update(status='completed' if result.returncode==0 else 'failed',returncode=result.returncode,
        wall_seconds=time.perf_counter()-started,finished_at=now(),log_sha256=sha(out/'logs'/(label+'.log')))
    write(out/'status.json',record)
    if label=='perception_raft_stereo' and result.returncode!=0:break
record.update(status='finished',finished_at=now());write(out/'status.json',record)
