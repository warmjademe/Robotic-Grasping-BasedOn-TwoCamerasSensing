#!/usr/bin/env python3
"""Development RQ experiment: immutable predictions, separate evaluation, no robot I/O.

Run on HuaShuo. Formal test and robot execution need separate validated gates.
All model outputs and failures are preserved. A successful process is never a grasp.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PROJECT = STUDY.parent
DATASET = STUDY/'datasets/industrial_workpieces_v1'
GRASP = PROJECT/'Source_Codes_Grasping_Modules'
PYTHON = GRASP/'baseline_graspgen/.venv-infer/bin/python'
SOURCE = HERE/'runs/development_perception72_20261011'
RUN_ID = 'development_rq_20261012'
RUN = HERE/'runs'/RUN_ID
STEREO = ['stereobm','sgm','foundationstereo','psmnet','raft_stereo','igev_stereo','gwcnet']
GRASPS = ['graspgen','6dof_graspnet','contact_graspnet','gpd','graspnet_baseline']
PAIRS = [(m,'graspgen') for m in STEREO] + [('foundationstereo',g) for g in GRASPS[1:]]


def read(p): return json.loads(Path(p).read_text())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); temp.replace(p)
def relative(p): return str(Path(p).relative_to(STUDY))
def episodes():
    result = [e for e in read(DATASET/'episodes.json') if e['split']=='development']
    assert len(result)==72 and len({e['episode_id'] for e in result})==72
    return result
def allowed():
    c=read(HERE/'task_control.json')
    if not c.get('launch_enabled') or c.get('study_scope')!='calibrated_RGB_stereo_only':
        raise RuntimeError('Launch is disabled or study scope differs')
def csv_write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    temp.replace(path)


def prepare():
    allowed(); RUN.mkdir(parents=True,exist_ok=False)
    selected=episodes(); predictions=[]; jobs=[]
    for method in STEREO:
        for e in selected:
            ep=e['episode_id'];p=SOURCE/'perception'/method/ep
            record=read(p/'perception.json')
            assert record['split']=='development' and record['method']==method
            assert record['geometry_truth_used'] is False and record['object_pose_used'] is False
            for name,digest in record['input_sha256'].items():
                assert sha(DATASET/'inputs'/ep/name)==digest,(ep,name,'input hash')
            assert sha(p/'prediction.npz')==record['prediction_sha256']
            predictions.append(dict(episode_id=ep,method=method,source=relative(p),
                prediction_sha256=record['prediction_sha256'],cloud_sha256=sha(p/'cloud.npz'),
                grasp_input_sha256=sha(p/'grasp_input.npz'),record_sha256=sha(p/'perception.json')))
    for e in selected:
        for method,grasp in PAIRS:
            jobs.append(dict(episode_id=e['episode_id'],perception=method,grasp=grasp,
                input=relative(SOURCE/'perception'/method/e['episode_id']/'grasp_input.npz'),
                output=relative(RUN/'grasps'/method/e['episode_id']/grasp),status='pending'))
    snap=RUN/'source_snapshot';snap.mkdir()
    for p in HERE.glob('*.py'): shutil.copy2(p,snap/p.name)
    shared=snap/'grasp_shared';shared.mkdir()
    for p in (GRASP/'shared').glob('*.py'): shutil.copy2(p,shared/p.name)
    for name in ['evaluation_protocol.json','fanuc_integration_requirements.json']:
        shutil.copy2(STUDY/'datasets'/name,snap/name)
    manifest=dict(schema_version=1,run_id=RUN_ID,created_at=now(),split='development',
        episodes=[e['episode_id'] for e in selected],dataset_manifest_sha256=sha(DATASET/'manifest.json'),
        perception_methods=STEREO,grasp_methods=GRASPS,pipeline_combinations=PAIRS,
        unique_combinations=11,expected_perception_results=504,expected_grasp_jobs=792,
        seed_policy='Development only: retain current backend seeds; not three-seed formal evidence',
        backend_seeds=dict(graspgen=0,**{'6dof_graspnet':0,'contact_graspnet':0,'graspnet_baseline':0,'gpd':20261011}),
        model_weights_updated=False,multiview_included=False,test_set_started=False,
        native_fanuc_executions=0,fanuc_grasp_success_rate=None,
        source_sha256={str(p.relative_to(snap)):sha(p) for p in snap.rglob('*') if p.is_file()})
    write(RUN/'manifest.json',manifest);write(RUN/'prediction_index.json',predictions)
    write(RUN/'jobs.json',jobs)
    write(RUN/'candidate_status.json',dict(status='prepared',completed=0,failed=0,expected=len(jobs)))
    write(RUN/'geometry_status.json',dict(status='prepared',completed=0,expected=len(predictions)))
    analyze()


def geometry():
    allowed()
    lock=open('/tmp/robotic_grasping_rq_geometry.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from evaluate_perception import evaluate
    index=read(RUN/'prediction_index.json');state=dict(status='running',pid=os.getpid(),started_at=now(),completed=0,failed=0,expected=len(index))
    write(RUN/'geometry_status.json',state)
    try:
        for item in index:
            allowed();method=item['method'];ep=item['episode_id']
            dest=STUDY/'RQ1/data/geometry'/method/(ep+'.json')
            if dest.exists(): raise FileExistsError(dest)
            pred=STUDY/item['source']
            assert sha(pred/'cloud.npz')==item['cloud_sha256']
            result=evaluate(DATASET,ep,pred,output=dest)
            result.update(prediction_cloud_sha256=item['cloud_sha256'],method=method,split='development',
                evaluator_sha256=sha(HERE/'evaluate_perception.py'),source=relative(pred))
            write(dest,result);state['completed']+=1;state['updated_at']=now()
            write(RUN/'geometry_status.json',state)
            if state['completed']%12==0:analyze()
        state.update(status='completed',finished_at=now());write(RUN/'geometry_status.json',state);analyze()
    except BaseException as exc:
        state.update(status='stopped',error=str(exc),finished_at=now());write(RUN/'geometry_status.json',state);raise


def candidates():
    allowed()
    lock=open('/tmp/robotic_grasping_grasp_eval.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=read(RUN/'jobs.json');state=dict(status='running',pid=os.getpid(),started_at=now(),completed=0,failed=0,expected=len(jobs))
    write(RUN/'candidate_status.json',state)
    child=None
    def stop(signum, frame): raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,stop)
    try:
        for n,job in enumerate(jobs):
            allowed();dest=STUDY/job['output'];source=STUDY/job['input']
            if job['status']!='pending' or dest.exists(): raise FileExistsError('Use a new run; no implicit retries')
            job.update(status='running',started_at=now(),input_sha256=sha(source))
            state['current']=dict(episode=job['episode_id'],perception=job['perception'],grasp=job['grasp'])
            write(RUN/'jobs.json',jobs);write(RUN/'candidate_status.json',state)
            log=RUN/'logs'/f'{n:04d}.log';log.parent.mkdir(exist_ok=True)
            cmd=[str(PYTHON),str(GRASP/'shared/infer.py'),'--method',job['grasp'],'--input',str(source),'--out',str(dest)]
            start=time.perf_counter()
            with log.open('x') as stream:
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try: rc=child.wait(timeout=900);status='completed' if rc==0 else 'model_error'
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    rc=child.returncode;status='timeout'
            child=None
            job.update(status=status,returncode=rc,wall_seconds=time.perf_counter()-start,
                finished_at=now(),log=relative(log),log_sha256=sha(log))
            if status=='completed':
                r=read(dest/'grasp_generation.json')
                assert r['input_npz_sha256']==job['input_sha256']
                assert r['output_sha256']==sha(dest/'grasps.npz')
                job.update(returned=r['returned'],valid_width_count=r['width_valid_count'],
                    result_sha256=sha(dest/'grasp_generation.json'))
                state['completed']+=1
            else:state['failed']+=1
            state['updated_at']=now();write(RUN/'jobs.json',jobs);write(RUN/'candidate_status.json',state)
            analyze();print(json.dumps(dict(index=n,**job)),flush=True)
        state.update(status='completed',finished_at=now(),current=None)
        write(RUN/'candidate_status.json',state);analyze()
    except BaseException as exc:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='stopped',error=str(exc),finished_at=now());write(RUN/'candidate_status.json',state);raise


def average(xs):
    xs=[x for x in xs if x is not None];return statistics.mean(xs) if xs else None
def quantile(xs,q):
    xs=sorted(x for x in xs if x is not None)
    if not xs:return None
    i=(len(xs)-1)*q;lo=int(i);hi=min(lo+1,len(xs)-1)
    return xs[lo]*(hi-i)+xs[hi]*(i-lo) if hi!=lo else xs[lo]


def analyze():
    raise SystemExit('Use publication/reproduce_tables.py --out NEW_DIRECTORY for the current three-RQ analysis.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','geometry','candidates','analyze'])
    args=p.parse_args();globals()[args.action]()
