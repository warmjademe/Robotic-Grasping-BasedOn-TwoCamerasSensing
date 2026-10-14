#!/usr/bin/env python3
"""Score a declared model/seed cohort from real records; missing results stay null."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

FAILURES={'perception_failure','model_error','no_candidates','width_invalid','ik_unreachable',
          'planning_failure','collision','grip_geometry_failure','lift_failure','hold_failure','timeout','human_intervention'}
TIMES=['perception_wall_s','grasp_wall_s','planning_wall_s','decision_wall_s','execution_sim_s','task_wall_s']


def score(episodes, records, method, seed, split, mode, evidence_root=None):
    expected={e['episode_id']:e for e in episodes if e['split']==split}
    groups=defaultdict(list);invalid=0
    for r in records:
        if r['method_id']!=method or r['seed']!=seed:continue
        assert r['episode_id'] in expected,'Unknown or wrong-split episode'
        assert r['success_mode']==mode,'Do not mix geometric/contact-physics results'
        if r['status']=='infrastructure_invalid':
            assert r.get('infrastructure_reason') and r.get('attempt_id')
            invalid+=1;continue
        assert r['status']=='completed' and type(r['grasp_success']) is bool
        assert r.get('evidence_files'),'Execution or algorithm failure evidence required'
        for f,h in r['evidence_files'].items():
            path=Path(f)
            assert not path.is_absolute() and '..' not in path.parts
            if evidence_root is not None:
                assert hashlib.sha256((evidence_root/path).read_bytes()).hexdigest()==h
        if r['grasp_success']:
            assert r['native_execution_performed'] is True and r['human_interventions']==0
            assert r['prohibited_collision'] is False and r['joint_limit_violation'] is False
            assert r['target_identity_correct'] and r['grip_geometry_valid']
            assert r['bottom_clearance_m']>=.1 and r['hold_sim_s']>=2
            assert r['failure_reason'] is None
            if mode=='contact_physics':
                assert r['contact_evaluator_validated'] and not r['target_rigidly_attached']
                assert r['relative_translation_drift_m']<=.002 and r['relative_rotation_drift_deg']<=5
        else:
            assert r['failure_reason'] in FAILURES
        for key in TIMES:
            value=r.get(key)
            assert value is None or (isinstance(value,(int,float)) and np.isfinite(value) and value>=0)
        groups[r['episode_id']].append(r)
    assert all(len(v)==1 for v in groups.values()),'Multiple valid outcomes for one task; refusing best-of retries'
    rows=[v[0] for v in groups.values()]
    by_object=defaultdict(list)
    for r in rows:by_object[expected[r['episode_id']]['object_id']].append(int(r['grasp_success']))
    by_family=defaultdict(list)
    for r in rows:by_family[expected[r['episode_id']]['family']].append(int(r['grasp_success']))
    values=list(by_family.values());sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    complete=len(rows)==len(expected)
    successes=sum(r['grasp_success'] for r in rows)
    result=dict(method_id=method,seed=seed,split=split,success_mode=mode,
        expected_tasks=len(expected),completed_valid_tasks=len(rows),successes=successes,
        infrastructure_invalid_attempts=invalid,coverage_complete=complete,
        official_success_rate=successes/len(rows) if complete and rows else None,
        observed_success_rate_completed=successes/len(rows) if rows else None,
        macro_object_success_rate=float(np.mean([np.mean(v) for v in by_object.values()])) if complete and by_object else None,
        family_cluster_bootstrap_ci95=None,independent_shape_families=len(values),
        uncertainty_limit='v1 has only four test shape families; bootstrap intervals are exploratory',
        missing_episode_ids=sorted(set(expected)-set(groups)),
        failure_counts={reason:sum(r.get('failure_reason')==reason for r in rows) for reason in sorted(FAILURES)},
        timing={},measurement_status='complete' if complete else 'not_started' if not rows else 'partial',
        evidence_scope='File hash and declared event validation; not proof that arbitrary supplied logs came from a physical robot')
    if complete and len(values)>=2:
        rng=np.random.default_rng(20261011);idx=rng.integers(0,len(values),(5000,len(values)))
        sampled=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
        result['family_cluster_bootstrap_ci95']=np.quantile(sampled,[.025,.975]).tolist()
    for key in TIMES:
        # Motion cycle times of successful tasks only; all-task wall clocks retain failures.
        data=[r[key] for r in rows if r.get(key) is not None and (key!='execution_sim_s' or r['grasp_success'])]
        result['timing'][key]=dict(n=len(data),median=float(np.median(data)) if data else None,
                                  p95=float(np.quantile(data,.95)) if data else None,
                                  scope='successful_tasks' if key=='execution_sim_s' else 'all_tasks_with_measurement')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--method',required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--split',choices=['development','test'],default='test')
    p.add_argument('--mode',choices=['geometric','contact_physics'],default='geometric');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--evidence-root',type=Path,help='Root of the relative evidence paths; defaults to the ledger directory')
    a=p.parse_args();assert not a.out.exists(),'Preserve previous reports'
    episodes=json.loads((a.dataset/'episodes.json').read_text())
    records=[json.loads(line) for line in a.records.read_text().splitlines() if line.strip()]
    report=score(episodes,records,a.method,a.seed,a.split,a.mode,a.evidence_root or a.records.parent)
    a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))
