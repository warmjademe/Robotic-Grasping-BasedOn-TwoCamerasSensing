"""One-time reconciliation of two diagnosed mesh-import interruptions."""
import json
from pathlib import Path
import subprocess
import time
from common import RUN, sha, write

JOBS=['nominal_task_0024_task_0049','nominal_task_0050_task_0083']
states=RUN/'ops/mujoco_jobs_v2'
archive=RUN/'ops/mesh_import_recovery_20261014'
archive.mkdir(exist_ok=True)
# Check both hosts, excluding shell command lines that only quote the filename.
check="""import pathlib
for d in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  args=(d/'cmdline').read_bytes().split(b'\\0')
  if any(a.endswith(b'/code/run_mujoco.py') for a in args):
   raise SystemExit('MuJoCo worker still active: '+d.name)
 except (FileNotFoundError,PermissionError,ProcessLookupError): pass
"""
subprocess.run(['python3','-c',check],check=True)
subprocess.run(['ssh','-o','BatchMode=yes','qyb@192.168.2.3',
    'python3 -c '+__import__('shlex').quote(check)],check=True)
for job in JOBS:
    path=states/(job+'.json'); state=json.loads(path.read_text())
    if state.get('recovery',{}).get('release_unexecuted'):continue
    backup=archive/(job+'.original.json')
    assert not backup.exists()
    write(backup,state)
    dest=Path(state['output'])
    if state['host']=='huashuo':
        dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az','qyb@192.168.2.3:'+str(dest)+'/',str(dest)+'/'],check=True)
    observed=[]
    for result in dest.glob('*/result.json'):
        data=json.loads(result.read_text());assert data['status']=='completed'
        observed.extend(r['task_id'] for r in data['rows'])
    assert set(observed)<=set(state['tasks']) and len(observed)==len(set(observed))
    log=Path(state['log']).read_text()
    assert 'AssertionError' in log and 'convex_parts' in log
    state.update(status='infrastructure_error',observed_tasks=sorted(observed),
        reconciled_unix=time.time(),recovery=dict(release_unexecuted=True,
        diagnosis='Trimesh hull processing discarded a near-degenerate face in two mounting_block_01 parts. '
                  'Retaining oriented Qhull simplices closes the mesh with unchanged vertices and volume.',
        runner_sha256=sha(RUN/'code/run_mujoco.py'),original_state=str(backup),
        pending_tasks=sorted(set(state['tasks'])-set(observed)),
        note='Existing physical outcomes are retained. Only tasks with no result are released for dispatch. '
             'Orphaned SSH return code and original end time are unknown, not inferred.'))
    write(path,state)
    print(json.dumps(dict(job=job,retained=observed,pending=state['recovery']['pending_tasks'])))
