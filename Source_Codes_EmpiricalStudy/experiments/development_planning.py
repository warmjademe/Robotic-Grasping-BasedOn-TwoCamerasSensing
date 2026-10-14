"""Preserve prediction-only development plans and failures for all 792 jobs."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from industrial_plan import plan

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SOURCE = HERE / 'runs/development_rq_20261012'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def run(output):
    lock = open('/tmp/rge-development-planning.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / 'source_snapshot'
    snapshots.mkdir()
    for name in ['industrial_plan.py', 'industrial_collision.py', 'development_planning.py']:
        shutil.copy2(HERE / name, snapshots / name)
    jobs = json.loads((SOURCE / 'jobs.json').read_text())
    assert len(jobs) == 792
    episodes = {e['episode_id']: e for e in json.loads(
        (STUDY / 'datasets/industrial_workpieces_v1/episodes.json').read_text())}
    write(output / 'manifest.json', dict(split='development', benchmark_eligible=False,
        uses_object_ground_truth=False, expected=len(jobs), source_jobs_sha256=sha(SOURCE / 'jobs.json'),
        source_sha256={p.name: sha(p) for p in snapshots.iterdir()},
        budget_seconds=30, candidate_limit=200,
        runtime_scope='Development timing; concurrent simulator commissioning may affect CPU load'))
    counts = Counter()
    state = dict(status='running', pid=os.getpid(), expected=len(jobs), completed=0,
                 started_at_unix=time.time())
    write(output / 'status.json', state)
    try:
        for index, job in enumerate(jobs):
            ep = episodes[job['episode_id']]
            assert ep['split'] == 'development'
            dest = output / job['perception'] / job['episode_id'] / job['grasp']
            dest.mkdir(parents=True, exist_ok=False)
            state.update(current={k: job[k] for k in ['episode_id', 'perception', 'grasp']})
            write(output / 'status.json', state)
            grasps = STUDY / job['output'] / 'grasps.npz'
            cloud = (STUDY / job['input']).with_name('cloud.npz')
            if job['status'] != 'completed':
                result = dict(status='upstream_model_error', native_execution_performed=False)
                write(dest / 'plan.json', result)
            elif job['returned'] == 0:
                result = dict(status='no_candidates', native_execution_performed=False)
                write(dest / 'plan.json', result)
            else:
                result = plan(cloud, grasps, ep['condition'], dest / 'plan.json')
            record = {k: job[k] for k in ['episode_id', 'perception', 'grasp']}
            record.update(status=result['status'], plan_path=str((dest / 'plan.json').relative_to(STUDY)),
                plan_sha256=sha(dest / 'plan.json'), wall_seconds=result.get('seconds'),
                selected_rank=(result.get('selected') or {}).get('rank'), index=index)
            with (output / 'index.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            counts[result['status']] += 1
            state.update(completed=index + 1, counts=dict(counts), updated_at_unix=time.time())
            write(output / 'status.json', state)
            print(json.dumps(record), flush=True)
        state.update(status='completed', current=None, finished_at_unix=time.time())
    except BaseException as error:
        state.update(status='stopped', error=repr(error), finished_at_unix=time.time())
        raise
    finally:
        write(output / 'status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
