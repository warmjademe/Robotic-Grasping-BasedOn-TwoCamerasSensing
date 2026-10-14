"""RQ2: audit saved GraspGen paths for seven stereo inputs; no inference or motion."""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics

STUDY=Path(__file__).resolve().parents[2]
ORDER=['stereobm','sgm','psmnet','gwcnet','raft_stereo','igev_stereo','foundationstereo']
NAMES=dict(zip(ORDER,['StereoBM','SGM','PSMNet','GwcNet','RAFT-Stereo','IGEV-Stereo','FoundationStereo']))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(rows,keys):
    output=[]
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        batch=[r for r in rows if tuple(r[k] for k in keys)==group]
        statuses=Counter(r['planning_status'] for r in batch)
        reasons=Counter(r['failure_stage'] for r in batch)
        row=dict(zip(keys,group))
        row.update(tasks=len(batch),grasp_model='graspgen',
            candidate_process_completed=sum(r['candidate_status']=='completed' for r in batch),
            candidate_nonempty=sum(r['returned_candidates']>0 for r in batch),
            path_found=statuses['sampled_path_found'],no_plan=statuses['no_plan'],
            no_candidates=statuses['no_candidates'],planning_timeout=statuses['planning_timeout'],
            path_found_rate=statuses['sampled_path_found']/len(batch),
            direction_or_width_only=reasons['direction_or_width_only'],
            pad_filter_only=reasons['pad_filter_only'],motion_checks_failed=reasons['motion_checks_failed'],
            planning_seconds_median=statistics.median(r['planning_seconds'] for r in batch),
            grasp_success_rate=None)
        assert row['path_found']+row['no_plan']+row['no_candidates']+row['planning_timeout']==len(batch)
        assert row['direction_or_width_only']+row['pad_filter_only']+row['motion_checks_failed']==row['no_plan']
        output.append(row)
    return output


def analyze(label):
    assert label and all(c.isalnum() or c in '_-' for c in label)
    assert not json.loads((STUDY/'experiments/task_control.json').read_text())['launch_enabled']
    data=STUDY/'RQ2/data'/label;results=STUDY/'RQ2/results'/label
    assert not data.exists() and not results.exists(), 'Use a new analysis label'
    data.mkdir(parents=True);results.mkdir(parents=True)
    sources={}
    def check(path,expected=None):
        value=sha(path)
        if expected is not None:assert value==expected,str(path)
        sources[str(path.relative_to(STUDY))]=value
        return path
    def read(path,expected=None):return json.loads(check(path,expected).read_text())
    source=STUDY/'experiments/runs/development_rq_20261012'
    jobs=[r for r in read(source/'jobs.json') if r['grasp']=='graspgen']
    assert len(jobs)==504 and {r['perception'] for r in jobs}==set(ORDER)
    manifest=read(source/'manifest.json');assert manifest['backend_seeds']['graspgen']==0
    plans_path=check(STUDY/'experiments/runs/development_planning_20261012/index.jsonl')
    plans=[json.loads(l) for l in plans_path.read_text().splitlines()]
    plans={(r['perception'],r['episode_id']):r for r in plans if r['grasp']=='graspgen'}
    assert len(plans)==504
    point_path=check(STUDY/'RQ1/data/development_pointclouds_20261012_v2/per_scene.csv')
    figure=read(STUDY/'RQ1/results/development_pointclouds_20261012_v2/figure_provenance.json')
    assert sha(point_path)==figure['source_sha256']
    with point_path.open() as f:clouds={(r['method'],r['episode_id']):r for r in csv.DictReader(f)}
    assert len(clouds)==504
    first={'width_invalid','approach_direction'}
    second=first|{'no_observed_pad_cross_section','observed_cross_section_outside_opening'}
    rows=[];rejection_rows=[];links={}
    for job in jobs:
        key=(job['perception'],job['episode_id']);item=plans[key];point=clouds[key]
        assert job['status']=='completed', 'Retain and explicitly handle process errors'
        check(STUDY/job['input'],job['input_sha256'])
        assert job['input_sha256']==point['grasp_input_sha256']
        generation=read(STUDY/job['output']/'grasp_generation.json',job['result_sha256'])
        check(STUDY/job['output']/'grasps.npz',generation['output_sha256'])
        assert generation['input_npz_sha256']==job['input_sha256']
        assert generation['returned']==job['returned']
        path=STUDY/item['plan_path'];plan=read(path,item['plan_sha256'])
        assert plan['status']==item['status']
        assert plan['cloud_sha256']==point['cloud_sha256']
        assert plan['grasps_sha256']==generation['output_sha256']
        assert plan['candidate_limit']==200 and plan['planning_budget_s']==30
        assert not plan['uses_object_ground_truth'] and not plan['native_execution_performed']
        rejected=Counter(r['reason'] for r in plan['rejected'])
        failure='none'
        if plan['status']=='no_plan':
            assert rejected
            failure='direction_or_width_only' if set(rejected)<=first else 'pad_filter_only' if set(rejected)<=second else 'motion_checks_failed'
        elif plan['status']!='sampled_path_found':failure=plan['status']
        if plan['status']=='sampled_path_found':
            assert plan['selected'] and {x['phase'] for x in plan['selected']['path']}=={'transit','approach','lift'}
        link=data/'plans'/job['perception']/(job['episode_id']+'.json')
        link.parent.mkdir(parents=True,exist_ok=True)
        link.symlink_to(os.path.relpath(path,link.parent))
        assert sha(link)==item['plan_sha256']
        links[str(link.relative_to(STUDY))]=os.readlink(link)
        candidate_link=data/'candidates'/job['perception']/job['episode_id']
        candidate_link.parent.mkdir(parents=True,exist_ok=True)
        candidate_link.symlink_to(os.path.relpath(STUDY/job['output'],candidate_link.parent),target_is_directory=True)
        links[str(candidate_link.relative_to(STUDY))]=os.readlink(candidate_link)
        row=dict(episode_id=job['episode_id'],perception=job['perception'],grasp='graspgen',seed=0,
            split='development',object_id=point['object_id'],family=point['family'],condition=point['condition'],
            candidate_status=job['status'],returned_candidates=job['returned'],valid_width_count=job['valid_width_count'],
            planning_status=plan['status'],path_found=plan['status']=='sampled_path_found',failure_stage=failure,
            selected_rank=item['selected_rank'],planning_seconds=plan['seconds'],grasp_cold_process_seconds=job['wall_seconds'],
            target_points=int(point['target_points']),surface_mae_mm=float(point['surface_mae_mm']),
            precision_2mm=float(point['precision_2mm']),coverage_2mm=float(point['coverage_2mm']),
            target_source_pixel_fraction=float(point['target_source_pixel_fraction']),
            input_source=job['input'],input_sha256=job['input_sha256'],candidate_source=job['output'],
            candidate_record_sha256=job['result_sha256'],plan_source=item['plan_path'],plan_sha256=item['plan_sha256'],
            rejection_counts_json=json.dumps(dict(rejected),sort_keys=True),
            native_execution_included=False,grasp_success=None)
        rows.append(row)
        for reason,count in rejected.items():
            rejection_rows.append(dict(episode_id=job['episode_id'],perception=job['perception'],
                planning_status=plan['status'],reason=reason,rejected_candidates=count,
                plan_source=item['plan_path'],plan_sha256=item['plan_sha256']))
    methods=summarize(rows,['perception']);conditions=summarize(rows,['perception','condition'])
    objects=summarize(rows,['perception','object_id'])
    assert all(r['tasks']==72 for r in methods)
    reference={r['episode_id']:r for r in rows if r['perception']=='foundationstereo'}
    paired=[]
    for method in ORDER:
        batch=[r for r in rows if r['perception']==method]
        both=sum(r['path_found'] and reference[r['episode_id']]['path_found'] for r in batch)
        only=sum(r['path_found'] and not reference[r['episode_id']]['path_found'] for r in batch)
        ref_only=sum(not r['path_found'] and reference[r['episode_id']]['path_found'] for r in batch)
        neither=len(batch)-both-only-ref_only
        paired.append(dict(perception=method,reference='foundationstereo',paired_scenes=len(batch),
            both_pass=both,only_method_pass=only,only_reference_pass=ref_only,neither_pass=neither,
            paired_rate_difference=(only-ref_only)/len(batch),inference='development_descriptive_no_p_values'))
    for name,values in [('per_scene.csv',rows),('by_method.csv',methods),('by_condition.csv',conditions),
                        ('by_object.csv',objects),('candidate_rejections.csv',rejection_rows),('paired_vs_foundationstereo.csv',paired)]:
        write_csv(data/name,values)
    summary=dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),research_question='固定 GraspGen 时，不同双目感知方法的抓取路径生成表现如何？',
        split='development',scene_count=72,method_count=7,task_count=len(rows),grasp_model='graspgen',
        counts=dict(Counter(r['planning_status'] for r in rows)),failure_stages=dict(Counter(r['failure_stage'] for r in rows)),
        primary_metric='sampled_path_found / all scheduled scene-method pairs',
        native_success_is_not_evaluated_here=True,inference_rerun=False,robot_commands_sent=False,
        source_sha256=sha(Path(__file__)),input_files=sources,raw_record_links=links,
        output_sha256={p.name:sha(p) for p in data.glob('*.csv')},
        scope='Frozen development candidate outputs and limited sampled planner; descriptive paired analysis')
    (data/'source_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(__file__,results/'analysis_source.py')
    table=['| 双目方法 | 有候选 | 路径检查通过 | 未通过 | 路径检查通过率 |','|---|---:|---:|---:|---:|']
    failure_table=['| 双目方法 | 朝向／开口筛选后无候选 | 接触区域筛选后无候选 | 进入运动检查后失败 |','|---|---:|---:|---:|']
    lookup={r['perception']:r for r in methods}
    for method in ORDER:
        r=lookup[method]
        table.append(f"| {NAMES[method]} | {r['candidate_nonempty']}/72 | {r['path_found']}/72 | {r['no_plan']} | {r['path_found_rate']:.1%} |")
        failure_table.append(f"| {NAMES[method]} | {r['direction_or_width_only']} | {r['pad_filter_only']} | {r['motion_checks_failed']} |")
    text=('# RQ2：固定 GraspGen 时，不同双目感知方法的抓取路径生成表现如何？\n\n'
        '读取已保存的 72 个开发场景 × 7 个双目方法，共 504 条 GraspGen 记录。使用相同夹爪、规划器和预算；不重新推理或执行机械臂。'
        '全部 504 次候选生成均正常完成且返回候选。\n\n'+'\n'.join(table)+'\n\n'
        '表中分母均为各方法全部 72 个场景。共 156 条通过、348 条未通过，本子集中没有空候选和规划超时。'
        '通过表示当前采样规划检查得到接近和抬升路径，最终抓取结果在下一阶段评价。\n\n'+'\n'.join(failure_table)+'\n\n'
        '失败阶段按任务达到的最深检查步骤互斥归类：候选朝向／开口筛选、目标点云接触区域筛选、运动检查。'
        '同一任务可能有多个候选分别在不同步骤被拒绝；逐候选原因另存在 candidate_rejections.csv，不能将其次数当作失败任务数。\n\n'
        '当前规划最多检查评分前 200 个候选，预算 30 s，只允许偏离竖直向下不超过 45°、开口 2–80 mm；'
        '检查初始位姿到预抓取的关节插值、120 mm 直线接近及 150 mm 直线抬升，使用 3 mm 点云体素。'
        '未进行绕障搜索；no_plan 表示这套配置未通过检查，不证明其他路线不存在。\n\n'
        'RQ1 指标按相同场景和点云哈希加入 per_scene.csv，可追溯路径结果对应的点数、精度、覆盖率及目标来源。'
        '方法差异同时包含深度估计和目标点云提取的差异。现有结果只支持这条固定 GraspGen 与规划配置下的比较，'
        '不据此将失败全部归因于抓取模型，也不把与点云指标的共同变化当作因果效应。\n\n'
        'by_condition.csv 和 by_object.csv 保留分条件／工件结果；paired_vs_foundationstereo.csv 按相同场景列出双方通过、'
        '仅一方通过和双方不通过的数量。当前只有两个开发形状族，报告描述性配对差异，不新增显著性检验或调参。\n')
    (results/'抓取路径生成分析.md').write_text(text)
    print(json.dumps(dict(tasks=len(rows),counts=summary['counts'],failure_stages=summary['failure_stages'],report=str(results/'抓取路径生成分析.md')),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',default='development_graspgen_20261012');analyze(p.parse_args().label)
