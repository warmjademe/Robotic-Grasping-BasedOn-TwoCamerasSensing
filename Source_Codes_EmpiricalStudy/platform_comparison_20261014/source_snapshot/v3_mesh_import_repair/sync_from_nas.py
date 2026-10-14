"""Finite experiment-result synchronization to the researcher's Mac workspace."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

RUN=Path(__file__).resolve().parents[1];STUDY=RUN.parent
REMOTE='/home/qyb/RESEARCH/Robotic_Grasping/Source_Codes_EmpiricalStudy/'

def sync():
    for name in ['results','analysis','calibration','ops','logs']:
        subprocess.run(['rsync','-az','--exclude=*.lock','SuJian_ubuntu_NAS:'+REMOTE+RUN.name+'/'+name+'/',str(RUN/name)+'/'],check=True,timeout=180)
    for rq in ['RQ2','RQ3','RQ4']:
        for kind in (['data','results'] if rq!='RQ4' else ['data']):
            rel=f'{rq}/{kind}/{RUN.name}/';dest=STUDY/rel;dest.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-az','SuJian_ubuntu_NAS:'+REMOTE+rel,str(dest)+'/'],check=True,timeout=180)
    progress=json.loads((RUN/'analysis/progress.json').read_text())
    verified=RUN/'analysis/input_verification.json'
    if not progress['complete'] or not verified.exists():return False
    assert json.loads(verified.read_text())['valid']
    manifest=json.loads((RUN/'inputs/manifest.json').read_text())
    changed=[rel for rel,expected in manifest['preserved_results_sha256'].items()
             if hashlib.sha256((STUDY/rel).read_bytes()).hexdigest()!=expected]
    assert not changed, 'Historical result changed'
    assert all(progress['physics_completed'].get(engine)==progress['expected_physics_per_engine'] for engine in ['isaac','mujoco'])
    (RUN/'ops/local_completion_verified.json').write_text(json.dumps(dict(time=datetime.datetime.now().astimezone().isoformat(),progress=progress,preserved_files=len(manifest['preserved_results_sha256']),valid=True),indent=2))
    endpoint=os.environ.get('BARK_ENDPOINT')
    marker=RUN/'ops/completion_notification.json'
    if endpoint and not marker.exists():
        body=json.dumps(dict(title='Franka 双平台开发实验执行完成',body='NAS/华硕调度批次的 Isaac Sim 与 MuJoCo 结果已收齐，RQ2、RQ3、RQ4 数据已同步到 Mac；旧结果校验通过。论文提纲已推送。',group='Robotic_Grasping')).encode()
        try:
            with urllib.request.urlopen(urllib.request.Request(endpoint,data=body,headers={'Content-Type':'application/json'}),timeout=15) as response:result=json.load(response)
            marker.write_text(json.dumps({'sent':result.get('code')==200,'time':datetime.datetime.now().astimezone().isoformat()}))
        except Exception as error:marker.write_text(json.dumps({'sent':False,'error_type':type(error).__name__}))
    return True

if __name__=='__main__':
    assert str(RUN).startswith('/Users/'), 'Run this coordinator on the Mac'
    lock=(RUN/'ops/local_sync.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        try:
            if sync():print('Completed and verified',flush=True);break
        except Exception as error:
            print(datetime.datetime.now().isoformat(),type(error).__name__,flush=True)
        time.sleep(60)
