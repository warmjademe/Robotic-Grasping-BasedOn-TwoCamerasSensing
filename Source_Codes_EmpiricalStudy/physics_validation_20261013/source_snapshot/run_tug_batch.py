"""Sequential GPU supervisor with calibration gate and immutable results."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time
RUN=Path(__file__).resolve().parents[1]
BASE=Path('/home/qyb/RESEARCH/baseline_graspgen')
def main():
    calibration=json.loads((RUN/'results/tug/calibration_v2/result.json').read_text())
    assert len(calibration['rows'])==3 and all(r['success']==r['expected'] for r in calibration['rows'])
    assert calibration['applied_closure_commands'] is not None
    m=json.loads((RUN/'inputs/manifest.json').read_text())
    env=dict(os.environ,LD_LIBRARY_PATH=str(BASE/'.sim-libs/lib'))
    jobs=[]
    for friction,label in [(1.,'nominal'),(.5,'friction_05')]:
        for group in sorted(m['groups']):
            destination=RUN/'results/tug'/label/group
            if (destination/'result.json').exists():
                jobs.append(dict(group=group,profile=label,status='completed',reused=True)); continue
            log=RUN/'logs'/('tug_'+label+'_'+group+'.log')
            cmd=[str(BASE/'.venv-sim/bin/python'),'-u',str(RUN/'code/run_graspdatagen.py'),
                 '--grasp_file',str(RUN/'inputs'/(group+'.yaml')),'--output',str(destination),
                 '--friction',str(friction)]
            state=dict(stage='tug',current=group,profile=label,completed=jobs,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (RUN/'status_tug.json').write_text(json.dumps(state,indent=2)+'\n')
            started=time.monotonic()
            with log.open('w') as f:
                code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=300).returncode
            job=dict(group=group,profile=label,exit_code=code,wall_seconds=time.monotonic()-started,
                status='completed' if code==0 and (destination/'result.json').exists() else 'infrastructure_error')
            jobs.append(job)
            if job['status']!='completed':raise RuntimeError(str(job))
            r=json.loads((destination/'result.json').read_text())
            assert len(r['rows'])==m['groups'][group]
            assert r['applied_closure_commands'] is not None
            print(label,group,len(r['rows']),sum(x['success'] for x in r['rows']),flush=True)
    (RUN/'status_tug.json').write_text(json.dumps(dict(stage='tug_complete',completed=jobs),indent=2)+'\n')
if __name__=='__main__':main()
