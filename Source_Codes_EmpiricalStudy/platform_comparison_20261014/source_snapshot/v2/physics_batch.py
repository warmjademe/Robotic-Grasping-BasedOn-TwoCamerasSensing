"""NAS coordinator: probe both hosts, prefer idle NAS, never duplicate jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from common import RUN, PROTOCOL, sha, write
from host_probe import probe

HUA='qyb@192.168.2.3'
BASE=Path('/home/qyb/RESEARCH/baseline_graspgen')

def host_probe(host):
    if host=='nas':return probe()
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',HUA,'python3',str(RUN/'code/host_probe.py')],capture_output=True,text=True,timeout=12)
    return json.loads(r.stdout) if r.returncode==0 else dict(cpu_ready=False,gpu_ready=False,error='probe_failed')

def calibrated(engine,profile):
    path=RUN/'calibration'/f'{engine}_{profile}_v2'
    files=[path/'result.json'] if engine=='isaac' else sorted(path.glob('*/result.json'))
    if not files or not all(p.exists() for p in files):return False
    rows=[r for p in files for r in json.loads(p.read_text())['rows']]
    by={r['task_id']:r for r in rows}
    if set(by)!=set(PROTOCOL['calibration']):return False
    return all(r['execution_success']==r['expected'] for r in rows) and not by['collision_blocked']['collision_free']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engine',choices=['isaac','mujoco'],required=True)
    p.add_argument('--attempt',default='v2');a=p.parse_args()
    lock=(RUN/'ops'/f'{a.engine}_coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads((RUN/'inputs/manifest.json').read_text())['tasks']
    profiles=list(PROTOCOL['physics']['profiles'])
    # Main calibration must be complete before processing model candidates.
    assert all(calibrated(a.engine,p) for p in profiles), 'Calibration gate failed'
    for key in ['code','protocols']:
        subprocess.run(['rsync','-az',str(RUN/key)+'/',HUA+':'+str(RUN/key)+'/'],check=True)
    states=RUN/'ops'/f'{a.engine}_jobs_{a.attempt}';states.mkdir(exist_ok=True)
    active={}
    def collect():
        for host,(process,state,log) in list(active.items()):
            rc=process.poll()
            if rc is None:continue
            log.close(); dest=Path(state['output'])
            if host=='huashuo':subprocess.run(['rsync','-az',HUA+':'+str(dest)+'/',str(dest)+'/'],check=True)
            files=[dest/'result.json'] if a.engine=='isaac' else list(dest.glob('*/result.json'))
            actual=[]
            for file in files:
                if file.exists():actual += [r['task_id'] for r in json.loads(file.read_text())['rows']]
            valid=rc==0 and sorted(actual)==sorted(state['tasks'])
            state.update(returncode=rc,finished_unix=time.time(),status='completed' if valid else 'infrastructure_error',
                observed_tasks=actual,wall_seconds=time.time()-state['started_unix'])
            write(states/(state['job_id']+'.json'),state);del active[host]
            if not valid:
                write(RUN/'ops'/f'{a.engine}_needs_diagnosis.json',state)
                raise RuntimeError('Infrastructure failure retained; stop new dispatch until diagnosed')
    while True:
        collect()
        scheduled={}
        for f in states.glob('*.json'):
            state=json.loads(f.read_text())
            for tid in state['tasks']:scheduled[(tid,state['profile'])]=state['status']
        selected=[]
        for t in tasks:
            file=RUN/'results/plans'/f"{t['task_id']}.json"
            if file.exists() and json.loads(file.read_text()).get('selected'):selected.append(t)
        waiting=[(t,profile) for profile in profiles for t in selected if (t['task_id'],profile) not in scheduled]
        if not waiting:
            if (RUN/'ops/planning_completed.json').exists() and not active:break
            time.sleep(10);continue
        for host in ['nas','huashuo']:
            if host in active:continue
            # MuJoCo is CPU based. Prefer a host without competing GPU training
            # as well, so benchmarking does not use an otherwise busy machine.
            try:resources=host_probe(host)
            except Exception:continue
            write(RUN/'ops'/f'resources_{host}.json',resources)
            ready=resources.get('gpu_ready') if a.engine=='isaac' else resources.get('cpu_experiment_ready') and resources.get('calibration_ready',{}).get('mujoco')
            if not ready:continue
            if not waiting:break
            first,profile=waiting[0]
            n=16 if a.engine=='isaac' else 8
            batch=[t for t,pf in waiting if pf==profile and t['object_id']==first['object_id']][:n]
            ids=[t['task_id'] for t in batch]
            job_id=f'{profile}_{ids[0]}_{ids[-1]}'
            dest=RUN/'results'/a.engine/a.attempt/job_id
            state=dict(job_id=job_id,engine=a.engine,host=host,tasks=ids,profile=profile,status='assigned',output=str(dest),
                started_unix=time.time(),resources=resources,protocol_sha256=sha(RUN/'protocols/protocol.json'))
            write(states/(job_id+'.json'),state)
            if host=='huashuo':
                subprocess.run(['rsync','-az',str(RUN/'results/plans')+'/',HUA+':'+str(RUN/'results/plans')+'/'],check=True)
            env=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','LD_LIBRARY_PATH='+str(BASE/'.sim-libs/lib')]
            python=BASE/('.venv-sim' if a.engine=='isaac' else '.venv-mujoco')/'bin/python'
            command=['timeout','1200']+env+[str(python),str(RUN/'code'/f'run_{a.engine}.py'),'--tasks',*ids,'--profile',profile,'--output',str(dest)]
            if host=='huashuo':command=['ssh','-o','BatchMode=yes',HUA,shlex.join(command)]
            logfile=RUN/'logs'/f'{a.engine}_{a.attempt}_{job_id}.log';log=logfile.open('wb')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',coordinator_child_pid=process.pid,log=str(logfile));write(states/(job_id+'.json'),state)
            active[host]=(process,state,log)
            waiting=[(t,pf) for t,pf in waiting if not (pf==profile and t['task_id'] in ids)]
        time.sleep(10)
    errors=[p.name for p in states.glob('*.json') if json.loads(p.read_text())['status']!='completed']
    write(RUN/'ops'/f'{a.engine}_completed.json',dict(completed_unix=time.time(),selected=len(selected),jobs=len(list(states.glob('*.json'))),infrastructure_errors=errors))
