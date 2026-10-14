"""Export paired platform evidence without treating pending tests as failures."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from common import RUN, STUDY, sha, write

def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    temp.replace(path)

def summarize():
    manifest=json.loads((RUN/'inputs/manifest.json').read_text())
    tasks=manifest['tasks'];execution={};batches=[]
    for engine in ['isaac','mujoco']:
        for file in (RUN/'results'/engine/'v2').rglob('result.json'):
            data=json.loads(file.read_text())
            assert data['status']=='completed'
            for row in data['rows']:
                key=(row['task_id'],engine,row['profile'])
                assert key not in execution, key
                execution[key]=(row,data,file)
        for file in (RUN/'ops'/f'{engine}_jobs_v2').glob('*.json'):
            data=json.loads(file.read_text())
            wall=data.get('wall_seconds')
            if wall is None and data.get('finished_unix'):wall=data['finished_unix']-data['started_unix']
            batches.append(dict(engine=engine,job_id=data['job_id'],host=data['host'],profile=data['profile'],
                tasks=len(data['tasks']),status=data['status'],started_unix=data['started_unix'],
                finished_unix=data.get('finished_unix'),dispatch_to_collection_wall_seconds=wall,
                amortized_dispatch_to_collection_seconds=(wall/len(data['tasks'])) if wall else None,
                timing_scope='Includes startup and polling delay up to 10 s; amortized value is throughput cost, not single-case latency',source=str(file.relative_to(STUDY))))
    rows=[]
    for task in tasks:
        pf=RUN/'results/plans'/f"{task['task_id']}.json"
        plan=json.loads(pf.read_text()) if pf.exists() else dict(status='planning_pending',selected=None)
        for engine in ['isaac','mujoco']:
            for profile in ['nominal','friction_05']:
                row={k:task[k] for k in ['task_id','episode_id','object_id','family','condition','perception','grasp']}
                row.update(rqs=';'.join(task['rqs']),platform='franka_'+engine,profile=profile,
                    candidate_count=task['candidate_count'],planning_status=plan['status'],planning_seconds=plan.get('seconds'),
                    planning_host=plan.get('host'),selected_rank=(plan.get('selected') or {}).get('rank'),
                    physics_status='pending' if plan.get('selected') else 'not_applicable',execution_success=None,
                    overall_task_success=None if plan['status'] in ['planning_pending','infrastructure_error'] or plan.get('selected') else False)
                key=(task['task_id'],engine,profile)
                if key in execution:
                    result,meta,file=execution[key]
                    row.update({k:v for k,v in result.items() if k not in ['expected']})
                    row.update(physics_status='completed',overall_task_success=result['execution_success'],
                        execution_host=meta['host'],execution_batch_seconds=meta['seconds'],
                        execution_batch_size=meta.get('parallelism',1),trace_file=str(file.with_name('trace.npz').relative_to(STUDY)),
                        result_file=str(file.relative_to(STUDY)),result_sha256=sha(file))
                rows.append(row)
    csv_write(RUN/'analysis/per_task.csv',rows)
    summaries=[]
    for rq,method in [('RQ2','perception'),('RQ3','grasp')]:
        filtered=[r for r in rows if rq in r['rqs'].split(';')]
        csv_write(STUDY/rq/'data/platform_comparison_20261014/per_task.csv',filtered)
        grouped=defaultdict(list)
        for row in filtered:grouped[(row['platform'],row['profile'],row[method])].append(row)
        summary=[]
        for (platform,profile,name),group in sorted(grouped.items()):
            pending=sum(r['overall_task_success'] is None for r in group)
            successes=sum(r['overall_task_success'] is True for r in group)
            summary.append(dict(rq=rq,platform=platform,profile=profile,method=name,tasks=len(group),
                selected_plans=sum(r['planning_status']=='sampled_path_found' for r in group),
                physics_completed=sum(r['physics_status']=='completed' for r in group),pending=pending,
                execution_successes=successes,overall_success_rate=successes/len(group) if pending==0 else None,
                result_status='complete' if pending==0 else 'in_progress'))
        csv_write(STUDY/rq/'results/platform_comparison_20261014/by_method.csv',summary);summaries+=summary
    csv_write(RUN/'analysis/by_rq_method.csv',summaries)
    csv_write(STUDY/'RQ4/data/platform_comparison_20261014/execution_batches.csv',batches)
    csv_write(STUDY/'RQ4/data/platform_comparison_20261014/planning.csv',[
        {k:r.get(k) for k in ['task_id','episode_id','perception','grasp','planning_status','planning_seconds','planning_host']}
        for r in rows if r['platform']=='franka_isaac' and r['profile']=='nominal'])
    status=dict(expected_tasks=len(tasks),plans=dict(Counter(r['planning_status'] for r in rows if r['platform']=='franka_isaac' and r['profile']=='nominal')),
        physics_completed=dict(Counter(k[1] for k in execution)),expected_physics_per_engine=sum(r['planning_status']=='sampled_path_found' for r in rows if r['platform']=='franka_isaac' and r['profile']=='nominal')*2,
        complete=all(r['overall_task_success'] is not None for r in rows),reference_platform='Preserved separately: physics_validation_20261013')
    write(RUN/'analysis/progress.json',status)
    print(json.dumps(status),flush=True)
    return status

if __name__=='__main__':summarize()
