#!/usr/bin/env python3
"""Reproduce three-RQ tables from recorded outcomes without running a simulator."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def truth(value):
    if value not in ('True', 'False', True, False):
        raise ValueError(f'Missing or invalid outcome: {value!r}')
    return value is True or value == 'True'


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reproduce(out):
    active = json.loads((STUDY/'datasets/active_dataset_72.json').read_text())
    scenes = set(active['episode_ids'])
    assert len(scenes) == 72
    manifest = json.loads((STUDY/'platform_comparison_20261014/inputs/manifest.json').read_text())
    tasks = manifest['tasks']
    assert len(tasks) == 792
    task_keys = {(t['episode_id'], t['perception'], t['grasp']) for t in tasks}
    assert len(task_keys) == 792 and {t[0] for t in task_keys} == scenes
    paths = [STUDY/'RQ1/data/development_pointclouds_20261012_v2/per_scene.csv',
             STUDY/'physics_validation_20261013/analysis/per_candidate.csv',
             STUDY/'platform_comparison_20261014/analysis/per_task.csv',
             STUDY/'platform_comparison_20261014/inputs/manifest.json']
    points, fanuc, franka = (read_csv(p) for p in paths[:3])
    assert len(points) == 504 and len(fanuc) == 412 and len(franka) == 3168
    assert len({(r['episode_id'], r['method']) for r in points}) == 504
    groups = defaultdict(list)
    for row in points:
        assert row['episode_id'] in scenes
        groups[row['method']].append(row)
    geometry = []
    for method, rows in sorted(groups.items()):
        assert len(rows) == 72
        record = dict(method=method, scenes=len(rows),
                      target_points_median=statistics.median(float(r['target_points']) for r in rows))
        for key in ('surface_mae_mm', 'surface_p95_mm', 'precision_2mm', 'coverage_2mm',
                    'target_depth_mae_mm', 'target_depth_valid_fraction', 'target_source_pixel_fraction'):
            values = [float(r[key]) for r in rows if r[key] not in ('', 'None', 'nan')]
            record[key+'_scene_mean'] = statistics.mean(values) if values else None
            record[key+'_valid_scenes'] = len(values)
        geometry.append(record)
    lookup = {}
    for r in fanuc:
        key = (r['episode_id'], r['perception'], r['grasp'], r['profile'])
        assert key not in lookup
        lookup[key] = r
        assert truth(r['execution_success']) == (truth(r['physics_lift_success']) and
               truth(r['collision_free']) and truth(r['approach_endpoint_reached']))
    planned = {(r['episode_id'], r['perception'], r['grasp']) for r in fanuc}
    assert len(planned) == 206
    physics_manifest = json.loads((STUDY/'physics_validation_20261013/inputs/manifest.json').read_text())
    assert planned == {(r['episode_id'], r['perception'], r['grasp']) for r in physics_manifest['records']}
    assert planned <= task_keys
    outcomes = []
    for task in tasks:
        key = (task['episode_id'], task['perception'], task['grasp'])
        for profile in ('nominal', 'friction_05'):
            record = lookup.get((*key, profile))
            assert (record is not None) == (key in planned)
            outcomes.append(dict(episode_id=task['episode_id'], perception=task['perception'],
                grasp=task['grasp'], condition=task['condition'], platform='fanuc_isaac',
                profile=profile, planned=key in planned, completed=record is not None,
                success=truth(record['execution_success']) if record else False))
    seen = set()
    for row in franka:
        key = (row['episode_id'], row['perception'], row['grasp'], row['platform'], row['profile'])
        assert key not in seen and key[:3] in task_keys
        seen.add(key)
        assert row['platform'] in ('franka_isaac', 'franka_mujoco')
        assert row['profile'] in ('nominal', 'friction_05')
        selected = row['planning_status'] == 'sampled_path_found'
        assert row['physics_status'] == ('completed' if selected else 'not_applicable')
        success = truth(row['overall_task_success'])
        if selected:
            assert success == (truth(row['physics_lift_success']) and truth(row['collision_free']) and
                               truth(row['approach_endpoint_reached']))
        else:
            assert not success
        outcomes.append(dict(episode_id=row['episode_id'], perception=row['perception'],
            grasp=row['grasp'], condition=row['condition'], platform=row['platform'],
            profile=row['profile'], planned=selected, completed=selected, success=success))
    assert sum(r['completed'] for r in outcomes) == 1244
    summary = []
    for rq, fixed, value, varying, nmethods in [('RQ2', 'grasp', 'graspgen', 'perception', 7),
                                               ('RQ3', 'perception', 'foundationstereo', 'grasp', 5)]:
        by = defaultdict(list)
        for row in outcomes:
            if row[fixed] == value:
                by[(row['platform'], row['profile'], row[varying], 'all')].append(row)
                by[(row['platform'], row['profile'], row[varying], row['condition'])].append(row)
        assert len(by) == 3*2*nmethods*4
        for (platform, profile, method, condition), rows in sorted(by.items()):
            n = 72 if condition == 'all' else 24
            assert len(rows) == n and len({r['episode_id'] for r in rows}) == n
            wins = sum(r['success'] for r in rows)
            plans = sum(r['planned'] for r in rows)
            summary.append(dict(rq=rq, platform=platform, profile=profile, method=method,
                condition=condition, tasks=n, selected_plans=plans, executed=plans,
                successes=wins, success_rate=wins/n,
                conditional_success_rate=wins/plans if plans else None))
    out.mkdir(parents=True, exist_ok=False)
    save_csv(out/'rq1_pointcloud_quality.csv', geometry)
    save_csv(out/'rq2_rq3_success.csv', summary)
    save_csv(out/'all_task_outcomes.csv', outcomes)
    result = dict(scenes=72, pointcloud_evaluations=504, candidate_tasks=792,
                  unique_full_arm_executions=1244, task_platform_profile_rows=len(outcomes),
                  input_sha256={str(p.relative_to(STUDY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='A new output directory')
    reproduce(parser.parse_args().out)
