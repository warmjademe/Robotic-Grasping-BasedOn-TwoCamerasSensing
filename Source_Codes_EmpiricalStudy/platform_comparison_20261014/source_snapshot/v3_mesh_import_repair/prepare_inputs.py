"""Freeze all 792 tasks and retain source hashes; no source result mutation."""
from pathlib import Path
import datetime
import hashlib
import json

RUN = Path(__file__).resolve().parents[1]
STUDY = RUN.parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    dest = RUN / 'inputs/manifest.json'
    if dest.exists(): raise FileExistsError(dest)
    jobs_path = STUDY / 'experiments/runs/development_rq_20261012/jobs.json'
    jobs = json.loads(jobs_path.read_text())
    episodes = {e['episode_id']: e for e in json.loads((STUDY / 'datasets/industrial_workpieces_v1/episodes.json').read_text())}
    assert len(jobs) == 792
    files = {}
    rows = []
    for i, job in enumerate(jobs):
        ep = episodes[job['episode_id']]
        assert ep['split'] == 'development'
        cloud = str(Path(job['input']).with_name('cloud.npz'))
        grasps = job['output'] + '/grasps.npz'
        for rel in [cloud, grasps]:
            if rel not in files: files[rel] = sha(STUDY / rel)
        report = job['output'] + '/grasp_generation.json'
        assert sha(STUDY / report) == job['result_sha256']
        files[report] = job['result_sha256']
        rqs = (['RQ2'] if job['grasp'] == 'graspgen' else []) + (['RQ3'] if job['perception'] == 'foundationstereo' else [])
        rows.append(dict(task_id=f'task_{i:04d}', episode_id=job['episode_id'], object_id=ep['object_id'],
            family=ep['family'], condition=ep['condition'], perception=job['perception'], grasp=job['grasp'], rqs=rqs,
            cloud=cloud, grasps=grasps, candidate_count=job['returned'], model_status=job['status'],
            inherited_candidate_wall_seconds=job['wall_seconds'], cloud_sha256=files[cloud], grasps_sha256=files[grasps]))
    assert sum('RQ2' in r['rqs'] for r in rows) == 504
    assert sum('RQ3' in r['rqs'] for r in rows) == 360
    old = {}
    for directory in ['physics_validation_20261013/analysis', 'physics_validation_20261013/results',
                      'RQ1/data/development_pointclouds_20261012_v2', 'RQ4/results']:
        for p in (STUDY / directory).rglob('*'):
            if p.is_file(): old[str(p.relative_to(STUDY))] = sha(p)
    manifest = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), tasks=rows,
        expected_tasks=792, source_sha256=files, preserved_results_sha256=old, jobs_sha256=sha(jobs_path),
        protocol_sha256=sha(RUN/'protocols/protocol.json'), ground_truth_used_in_planning=False)
    dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (RUN/'inputs/prediction_transfer_files.txt').write_text('\n'.join(sorted(files))+'\n')
    print(json.dumps(dict(tasks=len(rows), frozen_sources=len(files), preserved_results=len(old))))

if __name__ == '__main__': main()
