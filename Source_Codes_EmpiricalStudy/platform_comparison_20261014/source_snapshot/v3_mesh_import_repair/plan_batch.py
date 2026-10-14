"""Bounded CPU planning pool; each immutable task runs once."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import os
import socket
import time
from common import RUN, sha, write
from plan import plan

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    lock=(RUN/'ops/planning.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=json.loads((RUN/'inputs/manifest.json').read_text())['tasks']
    pending=[t for t in tasks if not (RUN/'results/plans'/f"{t['task_id']}.json").exists()]
    write(RUN/'ops/planning_assignment.json',dict(host=socket.gethostname(),workers=a.workers,
        protocol_sha256=sha(RUN/'protocols/protocol.json'),tasks=[t['task_id'] for t in pending],started_unix=time.time()))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(plan,t):t for t in pending}
        for future in as_completed(jobs):
            t=jobs[future]
            try:r=future.result()
            except Exception as e:
                r=dict(task_id=t['task_id'],status='infrastructure_error',error_type=type(e).__name__,error=str(e),selected=None)
            r['planning_parallelism']=a.workers
            write(RUN/'results/plans'/f"{t['task_id']}.json",r)
            print(json.dumps({k:r.get(k) for k in ['task_id','status','seconds']}),flush=True)
    write(RUN/'ops/planning_completed.json',dict(completed_unix=time.time(),tasks=len(tasks),host=socket.gethostname()))
