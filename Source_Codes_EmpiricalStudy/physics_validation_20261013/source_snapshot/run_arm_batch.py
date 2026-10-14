"""Run full-arm replay only after positive, negative and collision calibration."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
RUN=Path(__file__).resolve().parents[1];BASE=Path('/home/qyb/RESEARCH/baseline_graspgen')
def main():
    cal=json.loads((RUN/'results/arm/calibration_v4/result.json').read_text())
    assert all(r['execution_success']==r['expected'] for r in cal['rows'])
    assert next(r for r in cal['rows'] if r['case_id']=='collision_blocked')['prohibited_contact_steps']>0
    assert next(r for r in cal['rows'] if r['case_id']=='centered_box')['returned_to_origin_success']
    assert json.loads((RUN/'status_tug.json').read_text())['stage']=='tug_complete'
    manifest=json.loads((RUN/'inputs/manifest.json').read_text())
    snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for f in (RUN/'code').glob('*.py'):
        target=snapshot/f.name
        if target.exists():assert target.read_bytes()==f.read_bytes()
        else:target.write_bytes(f.read_bytes())
    jobs=[];env=dict(os.environ,LD_LIBRARY_PATH=str(BASE/'.sim-libs/lib'))
    for friction,label in [(1.,'nominal'),(.5,'friction_05')]:
        for group in sorted(manifest['groups']):
            dest=RUN/'results/arm'/label/group
            if (dest/'result.json').exists():
                jobs.append(dict(group=group,profile=label,status='completed',reused=True));continue
            state=dict(stage='full_arm',group=group,profile=label,completed=jobs,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (RUN/'status_arm.json').write_text(json.dumps(state,indent=2)+'\n')
            cmd=[str(BASE/'.venv-sim/bin/python'),'-u',str(RUN/'code/run_arm.py'),'--group',group,
                 '--output',str(dest),'--friction',str(friction)]
            started=time.monotonic()
            with (RUN/'logs'/('arm_'+label+'_'+group+'.log')).open('w') as f:
                rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600).returncode
            job=dict(group=group,profile=label,exit_code=rc,wall_seconds=time.monotonic()-started,
               status='completed' if rc==0 and (dest/'result.json').exists() else 'infrastructure_error')
            jobs.append(job)
            if job['status']!='completed':raise RuntimeError(str(job))
            result=json.loads((dest/'result.json').read_text())
            assert len(result['rows'])==manifest['groups'][group]
            print(label,group,len(result['rows']),sum(r['physics_lift_success'] for r in result['rows']),
                  sum(r['execution_success'] for r in result['rows']),flush=True)
    (RUN/'status_arm.json').write_text(json.dumps(dict(stage='full_arm_complete',completed=jobs),indent=2)+'\n')
if __name__=='__main__':main()
