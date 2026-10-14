"""Audit immutable inputs and recorded PhysX trajectories, then export paired tables."""
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

RUN = Path(__file__).resolve().parents[1]
STUDY = RUN.parent
OUT = RUN / 'analysis'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def counts(rows):
    return dict(selected=len(rows), old_geometry=sum(r['old_geometric_success'] for r in rows),
        tug=sum(r['tug_success'] for r in rows), lift=sum(r['physics_lift_success'] for r in rows),
        endpoint=sum(r['approach_endpoint_reached'] for r in rows),
        collision_free=sum(r['collision_free'] for r in rows),
        execution=sum(r['execution_success'] for r in rows),
        lift_and_return=sum(r['returned_to_origin_success'] for r in rows),
        execution_and_return=sum(r['execution_success'] and r['returned_to_origin_success'] for r in rows))

def main():
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((RUN/'inputs/manifest.json').read_text())
    for rel, digest in manifest['source_sha256'].items():
        assert sha(STUDY/rel) == digest, rel
    for name in ['tug','arm']:
        assert json.loads((RUN/f'status_{name}.json').read_text())['stage'].endswith('_complete')
    cal_t = json.loads((RUN/'results/tug/calibration_v2/result.json').read_text())
    cal_a = json.loads((RUN/'results/arm/calibration_v4/result.json').read_text())
    assert all(r['success'] == r['expected'] for r in cal_t['rows'])
    assert all(r['execution_success'] == r['expected'] for r in cal_a['rows'])
    assert next(r for r in cal_a['rows'] if r['case_id']=='collision_blocked')['prohibited_contact_steps'] > 0
    records = {r['case_id']:r for r in manifest['records']}
    assert len(records) == 206
    all_rows=[]; evidence={}; audits=[]; summaries={}; timings={}
    for profile in ['nominal', 'friction_05']:
        rows=[]
        for group, expected_count in manifest['groups'].items():
            arm_dir=RUN/'results/arm'/profile/group
            tug_dir=RUN/'results/tug'/profile/group
            arm=json.loads((arm_dir/'result.json').read_text())
            tug=json.loads((tug_dir/'result.json').read_text())
            assert arm['manifest_sha256']==sha(RUN/'inputs/manifest.json')
            assert arm['code_sha256']==sha(RUN/'source_snapshot/run_arm.py')
            assert tug['code_sha256']==sha(RUN/'source_snapshot/run_graspdatagen.py')
            assert not arm['object_attached'] and not arm['object_pose_written_after_initialization']
            assert len(arm['rows'])==len(tug['rows'])==expected_count
            trace=np.load(arm_dir/'trace.npz')
            assert np.isfinite(trace['object_pose']).all() and np.isfinite(trace['bodies']).all()
            assert np.allclose(np.diff(trace['time_s']),.02)
            hold=trace['phases']=='hold'; assert hold.sum()==100
            assert abs(trace['time_s'][-1]-19.5)<1e-6
            by_tug={r['case_id']:r for r in tug['rows']}
            contact_names=list(trace['contact_body_names'])
            finger_indices=[contact_names.index(n) for n in ['panda_leftfinger','panda_rightfinger']]
            housing_indices=[i for i,n in enumerate(contact_names) if n not in ['panda_leftfinger','panda_rightfinger','base_link']]
            hand_index=list(trace['body_names']).index('panda_hand')
            active=trace['phases']!='settle'
            for i,a in enumerate(arm['rows']):
                cid=a['case_id']; assert trace['case_ids'][i]==cid
                rec=records[cid]; t=by_tug[cid]
                assert np.isclose(a['closing_command_m'],rec['closing_command_m'])
                assert np.isclose(t['closing_command_m'],rec['closing_command_m'])
                mesh=trimesh.load(STUDY/rec['mesh_source'],force='mesh')
                pose=trace['object_pose'][:,i]
                rotation=Rotation.from_quat(pose[:,[4,5,6,3]]).as_matrix()
                z=np.array([(mesh.vertices@rot.T)[:,2].min()+p[2]-1.240 for rot,p in zip(rotation,pose)])
                assert abs(z[hold].min()-a['min_hold_bottom_clearance_m'])<1e-7
                assert bool(np.all(z[hold]>=.1))==a['physics_lift_success']
                assert a['execution_success']==(a['physics_lift_success'] and a['approach_endpoint_reached'] and a['collision_free'])
                non_target=trace['contact_forces'][:,i,finger_indices,:]-trace['finger_contact_matrix'][:,i,:,0,-1,:]
                sampled_bad=(np.linalg.norm(non_target,axis=-1).max(axis=1)>.5)|(np.linalg.norm(trace['contact_forces'][:,i,housing_indices,:],axis=-1).max(axis=1)>.5)
                if np.any(sampled_bad & active): assert not a['collision_free']
                if a['collision_free']: assert a['prohibited_contact_steps']==0
                collision_phases=sorted(set(trace['phases'][sampled_bad & active].tolist()))
                hand=trace['bodies'][hold,i,hand_index]
                hand_rotation=Rotation.from_quat(hand[:,[4,5,6,3]])
                relative=hand_rotation.inv().apply(pose[hold,:3]-hand[:,:3])
                relative_drift=float(np.linalg.norm(relative-relative[0],axis=1).max())
                target_forces=np.linalg.norm(trace['finger_contact_matrix'][hold,i,:,0,-1,:],axis=-1)
                contact_fraction=float(np.mean(np.all(target_forces>.05,axis=1)))
                rows.append(dict(profile=profile,**rec,tug_success=t['success'],**{k:v for k,v in a.items() if k not in ['case_id','expected','closing_command_m']},
                    hold_relative_translation_drift_m=relative_drift,hold_bilateral_contact_fraction=contact_fraction,
                    sampled_collision_phases=';'.join(collision_phases),
                    failure_lift=not a['physics_lift_success'],failure_endpoint=not a['approach_endpoint_reached'],failure_collision=not a['collision_free']))
            audits.append(dict(profile=profile,group=group,cases=expected_count,trace_frames=len(trace['time_s']),
                source_hashes=True,lift_recomputed=True,collision_samples_checked=True,closure_preserved=True))
            for folder in [arm_dir,tug_dir]:
                for file in [folder/'result.json',folder/'trace.npz']:
                    evidence[str(file.relative_to(RUN))]=sha(file)
        assert len(rows)==206 and len({r['case_id'] for r in rows})==206
        all_rows+=rows
        transition=Counter((r['old_geometric_success'],r['execution_success']) for r in rows)
        tug_transition=Counter((r['old_geometric_success'],r['tug_success']) for r in rows)
        summaries[profile]=dict(**counts(rows),old_to_execution={f'{int(a)}->{int(b)}':n for (a,b),n in sorted(transition.items())},
            old_to_tug={f'{int(a)}->{int(b)}':n for (a,b),n in sorted(tug_transition.items())},
            settling_translation_max_m=max(r['initial_settling_translation_m'] for r in rows),
            settling_translation_median_m=float(np.median([r['initial_settling_translation_m'] for r in rows])),
            execution_pass_max_hold_relative_drift_m=max((r['hold_relative_translation_drift_m'] for r in rows if r['execution_success']),default=None),
            execution_pass_min_bilateral_contact_fraction=min((r['hold_bilateral_contact_fraction'] for r in rows if r['execution_success']),default=None))
    # Only exported scalar columns; original hand matrices remain in manifest.
    scalar_rows=[{k:v for k,v in r.items() if k!='hand_pose_dataset_m'} for r in all_rows]
    write_csv(OUT/'per_candidate.csv',scalar_rows)
    rqrows=[]
    for rq,rel,fixed,variable in [
        ('RQ2','RQ2/results/native_validation_20261012/per_scene_native.csv',lambda r:r['grasp']=='graspgen','perception'),
        ('RQ3','RQ3/results/foundationstereo_comparison_20261012/per_scene.csv',lambda r:r['perception']=='foundationstereo','grasp')]:
        original=list(csv.DictReader((STUDY/rel).open()))
        for profile in summaries:
            for method in sorted({r[variable] for r in original}):
                subset=[r for r in all_rows if r['profile']==profile and fixed(r) and r[variable]==method]
                total=sum(r[variable]==method for r in original)
                assert total==72
                stat=counts(subset)
                rqrows.append(dict(rq=rq,profile=profile,method=method,all_scenes=total,**stat,
                    execution_over_selected=stat['execution']/len(subset) if subset else '',
                    execution_over_all_scenes=stat['execution']/total))
    write_csv(OUT/'by_rq_method.csv',rqrows)
    for rq, fixed in [('RQ2',lambda r:r['grasp']=='graspgen'),('RQ3',lambda r:r['perception']=='foundationstereo')]:
        target=STUDY/rq/'results/isaac_physics_20261013';target.mkdir(parents=True,exist_ok=True)
        write_csv(target/'per_candidate.csv',[r for r in scalar_rows if fixed(r)])
        write_csv(target/'by_method.csv',[r for r in rqrows if r['rq']==rq])
        (target/'README.md').write_text(
            f'# {rq}：Isaac Sim 物理复测\n\n'
            '读取旧计划选中的候选和原关节路径，未重新运行感知或候选选择。旧 ROBOGUIDE 结果保持原样。\n\n'
            '- `by_method.csv`：两种摩擦条件下的方法汇总，分别提供已选路径与全部 72 场景的分母。\n'
            '- `per_candidate.csv`：逐候选指标、旧几何结果、源计划路径和哈希。两个摩擦条件分别列出。\n'
            '- RQ2/RQ3 共享的 FoundationStereo + GraspGen 路径复用同一次物理执行，不能作为独立重复。\n\n'
            '[完整协议与原始记录](../../../physics_validation_20261013/README.md) · '
            '[物理复测报告](../../../physics_validation_20261013/analysis/物理复测报告.md)\n')
    condition_rows=[]
    for profile in summaries:
        for condition in sorted({r['condition'] for r in all_rows}):
            subset=[r for r in all_rows if r['profile']==profile and r['condition']==condition]
            condition_rows.append(dict(profile=profile,condition=condition,**counts(subset)))
    write_csv(OUT/'by_condition.csv',condition_rows)
    for stage in ['tug','arm']:
        jobs=json.loads((RUN/f'status_{stage}.json').read_text())['completed']
        assert len(jobs)==16 and all(j['status']=='completed' for j in jobs)
        timings[stage]=dict(batch_jobs=len(jobs),subprocess_wall_seconds=sum(j.get('wall_seconds',0) for j in jobs),
            note='Includes simulator startup; groups run candidates concurrently. Not robot cycle time.')
    contact_audits=[json.loads((RUN/'diagnostics'/name).read_text()) for name in ['contact_filter_audit.json','contact_filter_audit_friction_05.json']]
    assert all(r['status']=='passed' and r['unchanged_decisions'] for r in contact_audits)
    report=dict(scope='206 frozen selected development candidates; 792 original pipeline tasks; no new perception/inference',
        profiles=summaries,by_rq_method=rqrows,by_condition=condition_rows,timing=timings,
        verification=audits,contact_filter_audits=contact_audits,source_files_verified=len(manifest['source_sha256']),evidence_sha256=evidence)
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# 既有候选的物理复测结果','',
        '206 个不同的既有选中候选，在两组摩擦设置下各完成一次孤立夹爪测试和完整机械臂执行。共 824 次物理试验，不含校准。没有更换候选、重算点云或重新训练。',
        '', '| 条件 | 候选数 | 旧几何通过 | 抗扰动夹持通过 | 物理抬升保持 | 无禁止碰撞且到位的执行 | 执行通过并放回 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for profile,s in summaries.items():
        lines.append(f"| {profile} | {s['selected']} | {s['old_geometry']} | {s['tug']} | {s['lift']} | {s['execution']} | {s['execution_and_return']} |")
    lines += ['', '这些列评价不同条件，不是逐列筛选的漏斗。抗扰动测试悬空、无重力并向多个方向拉拽；完整机械臂测试有桌面、重力和动态接触，接近时可能推动或重新居中工件。',
        '', '默认条件下，旧几何失败中有 81 条通过物理执行，旧几何通过中有 13 条物理执行失败，双方共同通过 20 条。变化包含接触动力学、驱动与碰撞表示的共同影响。全部 206 条都到达预定抓取位姿，这一批的物理失败没有被归类为机械臂末端不到位。',
        '', '事后诊断检查发现：两组条件的执行通过记录，在 2 s 保持期的每个保存帧均有双侧夹指接触（每侧合力大于 0.05 N）。工件相对夹爪的最大位置变化分别为 3.71 mm 和 9.46 mm。此项用于核对是否仍被夹持，没有据此修改预先规定的成功阈值。',
        '', '执行通过要求抬升保持、接近到位以及从接近到放回全过程无禁止接触。放回误差另行记录。禁止接触检测使用非夹指连杆接触力与夹指非目标物体接触合力，阈值为 0.5 N；相反方向的接触力可能互相抵消，当前并非逐接触点的完整碰撞审计。',
        '', '## RQ2：固定 GraspGen','', '| 双目方法 | 已选路径 | 物理夹持 | 抬升保持 | 执行通过 | 执行/全部72场景 |','|---|---:|---:|---:|---:|---:|']
    for r in rqrows:
        if r['rq']=='RQ2' and r['profile']=='nominal':
            lines.append(f"| {r['method']} | {r['selected']} | {r['tug']} | {r['lift']} | {r['execution']} | {r['execution']}/72 |")
    lines+=['','## RQ3：固定 FoundationStereo','', '| 抓取模型 | 已选路径 | 物理夹持 | 抬升保持 | 执行通过 | 执行/全部72场景 |','|---|---:|---:|---:|---:|---:|']
    for r in rqrows:
        if r['rq']=='RQ3' and r['profile']=='nominal':
            lines.append(f"| {r['method']} | {r['selected']} | {r['tug']} | {r['lift']} | {r['execution']} | {r['execution']}/72 |")
    lines+=['','RQ2 与 RQ3 共享的 51 个候选只仿真一次。未找到路径的原任务在“执行/全部场景”中为流程未成功，而非物理夹持失败。',
        '', '## 单件与多件场景','', '| 条件 | 已选路径 | 抗扰动夹持 | 抬升保持 | 执行通过 |','|---|---:|---:|---:|---:|']
    for r in condition_rows:
        if r['profile']=='nominal':lines.append(f"| {r['condition']} | {r['selected']} | {r['tug']} | {r['lift']} | {r['execution']} |")
    lines += ['', '## 证据与解释边界','',
        f"- 已核验 {len(manifest['source_sha256'])} 个历史来源文件的 SHA-256，所有 412 条完整机械臂记录均从原始轨迹重新计算抬升高度。",
        '- 正例、夹空、不闭合和障碍物阻挡校准均通过；早期失败校准及日志保留，不纳入结果。',
        '- 另有 42 次接触过滤器独立复核。删除未使用的静态过滤槽后，两种材料条件的整组轨迹、关节值、接触力和结果逐值相同；当前执行代码已去掉该 SDK 错误提示，正式批次源码快照保留。',
        '- 物理模拟 250 Hz，轨迹保存 50 Hz，2 s 保持对应 100 帧；碰撞门禁在全部物理步计算。',
        '- 质量为每件 0.1 kg；材料参数、关节驱动和碰撞网格是当前仿真设置，未由真机辨识。没有关闭重力、挂接物体或在初始化后改写物体位姿。',
        '- 材料读回复核确认完整机械臂的 10 个机器人碰撞形状静/动摩擦分别为 1.0/1.0 和 0.5/0.5，恢复系数均为 0。孤立夹爪阶段保留作者夹爪 USD 材料，仅显式改变对象材料；两阶段测试负载和材料范围有差别。',
        '- 使用官方 M-20iD/25 几何/关节描述与 PhysX 驱动，未模拟 FANUC 原生控制器。固定各阶段时长合计 19.5 s，不能作为真实机械臂节拍。',
        '- 所有结果属于已观察开发集的配对诊断。旧几何判定、GraspDataGen 夹持和完整机械臂结果分别保存，不能以总数变化单独归因于旧判定代码。',
        '- 输入与逐候选指标见 inputs/manifest.json、analysis/per_candidate.csv；模型分组见 analysis/by_rq_method.csv；原始轨迹与接触记录见 results/。',
        '', '复核命令：使用华硕 .venv-infer/bin/python 运行 code/analyze_results.py。']
    (OUT/'物理复测报告.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(profiles=summaries,timing=timings,verified_sources=len(manifest['source_sha256'])),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
