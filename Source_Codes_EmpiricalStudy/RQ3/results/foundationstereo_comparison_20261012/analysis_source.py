"""Fixed-FoundationStereo comparison, installed after RQ2 native completion."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import sys

STUDY=Path(__file__).resolve().parents[2]
MODELS=['graspgen','6dof_graspnet','contact_graspnet','graspnet_baseline','gpd']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csv_write(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare_data():
    assert read(STUDY/'RQ2/data/native_validation_20261012/status.json')['all_paths_validated']
    assert read(STUDY/'RQ2/results/native_validation_20261012/analysis_manifest.json')['complete']
    data=STUDY/'RQ3/data/development_foundationstereo_20261012';data.mkdir(parents=True,exist_ok=False)
    source=STUDY/'experiments/runs/development_rq_20261012'
    jobs=[r for r in read(source/'jobs.json') if r['perception']=='foundationstereo']
    assert len(jobs)==360 and {r['grasp'] for r in jobs}==set(MODELS)
    seeds=read(source/'manifest.json')['backend_seeds']
    index=STUDY/'experiments/runs/development_planning_20261012/index.jsonl'
    plans={(r['episode_id'],r['grasp']):r for r in map(json.loads,index.read_text().splitlines()) if r['perception']=='foundationstereo'}
    clouds={(r['episode_id']):r for r in csv.DictReader((STUDY/'RQ1/data/development_pointclouds_20261012_v2/per_scene.csv').open()) if r['method']=='foundationstereo'}
    checked={str(index.relative_to(STUDY)):sha(index),str((source/'jobs.json').relative_to(STUDY)):sha(source/'jobs.json'),
             str((source/'manifest.json').relative_to(STUDY)):sha(source/'manifest.json'),
             'RQ1/data/development_pointclouds_20261012_v2/per_scene.csv':sha(STUDY/'RQ1/data/development_pointclouds_20261012_v2/per_scene.csv')}
    def check(p,expected=None):
        value=sha(p)
        if expected is not None:assert value==expected,str(p)
        checked[str(p.relative_to(STUDY))]=value
    rows=[];rejections=[]
    for job in jobs:
        assert job['status']=='completed'
        point=clouds[job['episode_id']];item=plans[(job['episode_id'],job['grasp'])]
        check(STUDY/job['input'],job['input_sha256']);assert job['input_sha256']==point['grasp_input_sha256']
        candidate=STUDY/job['output']/'grasp_generation.json';check(candidate,job['result_sha256'])
        generation=read(candidate);check(STUDY/job['output']/'grasps.npz',generation['output_sha256'])
        assert generation['input_npz_sha256']==job['input_sha256'] and generation['returned']==job['returned']
        path=STUDY/item['plan_path'];check(path,item['plan_sha256']);plan=read(path)
        assert plan['status']==item['status'] and not plan['native_execution_performed']
        if plan['status']!='no_candidates':
            assert plan['uses_object_ground_truth'] is False and plan['candidate_limit']==200 and plan['planning_budget_s']==30
            assert plan['cloud_sha256']==point['cloud_sha256'] and plan['grasps_sha256']==generation['output_sha256']
        else:assert job['returned']==0
        rejected=Counter(r['reason'] for r in plan.get('rejected',[]))
        row=dict(episode_id=job['episode_id'],perception='foundationstereo',grasp=job['grasp'],seed=seeds[job['grasp']],split='development',
            object_id=point['object_id'],family=point['family'],condition=point['condition'],candidate_status=job['status'],
            returned_candidates=job['returned'],valid_width_count=job['valid_width_count'],planning_status=plan['status'],
            path_found=plan['status']=='sampled_path_found',selected_rank=item.get('selected_rank'),
            planning_seconds=plan.get('seconds'),grasp_cold_process_seconds=job['wall_seconds'],
            input_source=job['input'],input_sha256=job['input_sha256'],candidate_source=job['output'],
            candidate_record_sha256=job['result_sha256'],plan_source=item['plan_path'],plan_sha256=item['plan_sha256'],
            rejection_counts_json=json.dumps(dict(rejected),sort_keys=True))
        rows.append(row)
        for reason,count in rejected.items():rejections.append(dict(episode_id=job['episode_id'],grasp=job['grasp'],reason=reason,rejected_candidates=count))
        for folder,target in [('plans',path),('candidates',STUDY/job['output'])]:
            link=data/folder/job['grasp']/(job['episode_id']+('.json' if folder=='plans' else ''))
            link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(os.path.relpath(target,link.parent),target_is_directory=folder=='candidates')
    assert sum(r['path_found'] for r in rows)==101
    csv_write(data/'per_scene.csv',rows);csv_write(data/'candidate_rejections.csv',rejections)
    write(data/'source_manifest.json',dict(fixed_perception='foundationstereo',tasks=360,models=MODELS,seeds=seeds,source_sha256=checked,per_scene_sha256=sha(data/'per_scene.csv')))
    summarize()


def summarize():
    data=STUDY/'RQ3/data/development_foundationstereo_20261012';result=STUDY/'RQ3/results/foundationstereo_comparison_20261012';result.mkdir(parents=True,exist_ok=True)
    path=data/'per_scene.csv';assert sha(path)==read(data/'source_manifest.json')['per_scene_sha256']
    rows=list(csv.DictReader(path.open()));run=STUDY/'RQ3/data/native_validation_20261012'
    nr={}
    if (run/'execution_records.csv').exists():
        snapshot=(run/'execution_records.csv').read_bytes();(result/'native_records_snapshot.csv').write_bytes(snapshot)
        nr={(r['episode_id'],r['grasp']):r for r in csv.DictReader(io.StringIO(snapshot.decode()))}
    sys.path.insert(0,str(STUDY/'RQ2/code'))
    from run_native_validation import verify_execution
    for row in rows:
        native=nr.get((row['episode_id'],row['grasp']));outcome={}
        if native and native['status'] in ['completed','completed_reused']:
            source=STUDY/native['execution_result_source'];assert sha(source)==native['execution_result_sha256']
            outcome=verify_execution(source,row['plan_sha256'])
            assert outcome['approach_monitor']['version']=='v10'
        row.update(native_status=native['status'] if native else ('pending' if row['path_found']=='True' else 'not_dispatched_no_path'),
            grasp_success=outcome.get('grasp_success'),failure_reason=outcome.get('failure_reason'),
            capture_failure_reason=outcome.get('capture',{}).get('failure_reason'),
            capture_json=json.dumps(outcome.get('capture'),sort_keys=True),
            native_execution_included=bool(outcome),worker=outcome.get('worker_id'),
            execution_sim_s=outcome.get('execution_sim_s'),dispatch_to_outcome_wall_s=outcome.get('measured_dispatch_to_outcome_wall_s'),
            native_source=outcome.get('execution_result_source'),native_source_sha256=outcome.get('execution_result_sha256'))
    summary=[]
    for model in MODELS:
        part=[r for r in rows if r['grasp']==model];assert len(part)==72
        pcounts=Counter(r['planning_status'] for r in part);ncounts=Counter(r['native_status'] for r in part)
        valid=[r for r in part if r['native_execution_included']];success=sum(r['grasp_success'] is True for r in valid)
        paths=pcounts['sampled_path_found'];complete=len(valid)==paths
        summary.append(dict(grasp=model,scenes=72,candidate_completed=72,nonempty=sum(int(r['returned_candidates'])>0 for r in part),
            candidate_count_median=statistics.median(int(r['returned_candidates']) for r in part),paths=paths,
            no_plan=pcounts['no_plan'],no_candidates=pcounts['no_candidates'],planning_timeout=pcounts['planning_timeout'],
            native_valid=len(valid),native_success=success,native_failure=len(valid)-success,
            pending=ncounts['pending'],running=ncounts['running'],infrastructure_invalid=ncounts['infrastructure_invalid'],
            all_paths_validated=complete,conditional_native_success_rate=success/len(valid) if complete and valid else None,
            end_to_end_success_rate=success/72 if complete else None,
            generation_cold_process_s_median=statistics.median(float(r['grasp_cold_process_seconds']) for r in part),
            execution_sim_s_median=statistics.median(r['execution_sim_s'] for r in valid) if valid else None))
    csv_write(result/'per_scene.csv',rows);csv_write(result/'by_model.csv',summary)
    if all(m['all_paths_validated'] for m in summary) and (run/'queue.sqlite').exists():
        with sqlite3.connect(f'file:{run}/queue.sqlite?mode=ro',uri=True) as source_db:
            with sqlite3.connect(result/'queue_snapshot.sqlite') as destination_db:source_db.backup(destination_db)
    report=['# RQ3：固定 FoundationStereo 时，不同抓取模型的抓取表现如何？','',
      '同一批 72 个开发场景、相同 FoundationStereo 点云输入，比较五个已实现抓取模块。沿用保存的候选和规划预算，不重新训练或选择候选。ASGrasp 不纳入本比较。','',
      '| 抓取模型 | 有候选 | 通过规划 | 有效原生执行 | 几何抓取成功 | 执行失败 | 待执行／执行中 | 环境异常 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in summary:report.append(f"| {m['grasp']} | {m['nonempty']}/72 | {m['paths']}/72 | {m['native_valid']} | {m['native_success']} | {m['native_failure']} | {m['pending']+m['running']} | {m['infrastructure_invalid']} |")
    report+=['','候选生成进程正常退出与生成非空候选分别统计；未找到路径、空候选和规划超时保留在全部 72 场景的端到端分母内。有效执行只统计通过证据核验的原生结果；环境异常和未执行项单列。', '',
      'GraspGen 的 FoundationStereo 子集与 RQ2 完全相同，复用经过哈希核验的执行证据，不作为新增独立样本。其他模块也按计划哈希复用已有完整执行记录，保留所有算法失败。','',
      '共同规划最多检查前 200 个候选，预算 30 s，夹爪开口 2–80 mm，朝向偏离竖直向下不超过 45°，采用固定接近／抬升与关节插值路线，无绕障搜索。各方法输出的原始候选数、时间和随机种子保留，因此候选总数不作为抓取质量排名。','',
      '仿真采用读回型号 ARC Mate 120iD、双指平行夹爪和相同场景。成功为双侧几何夹取后条件刚性附着，原生抬升底部至少 100 mm、保持至少 2 s，并无禁止碰撞；不模拟接触力和摩擦，精确 M-20iD/25 型号未核实。','',
      '本研究为开发集描述性比较，覆盖两个开发形状族。条件成功率和端到端成功率在该方法所有计划完成验证后计算；墙钟推理耗时与仿真运动耗时分别保留。','']
    (result/'抓取模型比较.md').write_text('\n'.join(report))
    shutil.copy2(__file__,result/'analysis_source.py')
    write(result/'analysis_manifest.json',dict(complete=all(m['all_paths_validated'] for m in summary),source_sha256=sha(path),
          expected_paths=101,models=summary,output_sha256={p.name:sha(p) for p in result.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','summarize']);a=p.parse_args()
    prepare_data() if a.action=='prepare' else summarize()
