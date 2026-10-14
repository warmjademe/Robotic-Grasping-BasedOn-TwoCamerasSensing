"""Development planner using predictions and fixed installation geometry only.

Plans are diagnostic until native trajectory/collision/clock calibration passes.
No evaluation-only object mesh, pose, depth or target mask is read here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from industrial_collision import RobotCollision, DATASET_TO_ROBOT, FLANGE_TO_HAND
from kinematics import from_wpr, to_wpr, solve, fk

HOME_POSE = [800, 160, 1540, 180, 0, 0]
HOME_SEED = np.deg2rad([11, -7, 10, 0, -100, -11])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan(cloud_path, grasp_path, condition, output, budget_s=30, candidate_limit=200):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Keep prior planning results')
    started = time.perf_counter()
    with np.load(cloud_path, allow_pickle=False) as data:
        target = data['target_points_m'].astype(float)
        scene = data['scene_points_m'].astype(float)
    with np.load(grasp_path, allow_pickle=False) as data:
        poses, scores = data['T_world_grasp'], data['score']
        widths, valid = data['jaw_width_m'], data['gripper_width_valid']
    result = dict(status='no_plan', condition=condition, uses_object_ground_truth=False,
        cloud_sha256=sha(cloud_path), grasps_sha256=sha(grasp_path),
        candidate_limit=candidate_limit, planning_budget_s=budget_s,
        native_execution_performed=False, full_trajectory_collision_validated=False,
        rejected=[], selected=None)
    qhome, e = solve(from_wpr(HOME_POSE), HOME_SEED)
    assert e[0] < .2 and e[1] < .1
    collision = RobotCollision(condition, scene_points=scene)
    result['home_check'] = collision.check(qhome, .08)
    if result['home_check']['collision']:
        result['status'] = 'home_collision'
    else:
        for rank in np.argsort(-scores, kind='stable')[:candidate_limit]:
            rank = int(rank)
            if time.perf_counter() - started > budget_s:
                result['status'] = 'planning_timeout'
                break
            H, opening = poses[rank].astype(float), float(widths[rank])
            reason = None
            if not valid[rank] or not np.isfinite(opening) or not .002 <= opening <= .08:
                reason = 'width_invalid'
            elif H[2, 2] > -np.cos(np.deg2rad(45)):
                reason = 'approach_direction'
            if reason:
                result['rejected'].append(dict(rank=rank, reason=reason))
                continue
            p = (target - H[:3, 3]) @ H[:3, :3]
            band = (np.abs(p[:, 0]) < .010) & (p[:, 2] > .090) & (p[:, 2] < .113)
            if band.sum() < 10:
                reason = 'no_observed_pad_cross_section'
            else:
                low, high = np.quantile(p[band, 1], [.01, .99])
                if not low < 0 < high or max(abs(low), abs(high)) > opening / 2:
                    reason = 'observed_cross_section_outside_opening'
            if reason:
                result['rejected'].append(dict(rank=rank, reason=reason))
                continue
            flange = DATASET_TO_ROBOT @ H @ np.linalg.inv(FLANGE_TO_HAND)
            pre = flange.copy()
            pre[:3, 3] -= .12 * (DATASET_TO_ROBOT @ H)[:3, 2]
            lift = flange.copy()
            lift[2, 3] += .15
            q, error = solve(pre, qhome)
            if error[0] > .2 or error[1] > .1:
                result['rejected'].append(dict(rank=rank, reason='pregrasp_ik'))
                continue
            path = []
            # Record all joint interpolation samples, including home-to-pregrasp.
            count = max(2, int(np.ceil(np.rad2deg(np.max(abs(q - qhome))))) + 1)
            for f in np.linspace(0, 1, count):
                if time.perf_counter() - started >= budget_s:
                    reason = 'planning_timeout'
                    break
                waypoint = qhome * (1 - f) + q * f
                check = collision.check(waypoint, opening)
                if check['collision']:
                    reason = 'transit_collision'
                    break
                path.append(dict(phase='transit', joints_deg=np.rad2deg(waypoint).tolist(),
                                 flange_xyz_wpr=to_wpr(fk(waypoint)), opening_m=opening))
            if not reason:
                for phase, start, end in [('approach', pre, flange), ('lift', flange, lift)]:
                    count = int(np.ceil(np.linalg.norm(end[:3, 3] - start[:3, 3]) / .005)) + 1
                    for f in np.linspace(0, 1, count):
                        if time.perf_counter() - started >= budget_s:
                            reason = 'planning_timeout'
                            break
                        T = start.copy()
                        T[:3, 3] = (1 - f) * start[:3, 3] + f * end[:3, 3]
                        q, error = solve(T, q)
                        if error[0] > .2 or error[1] > .1:
                            reason = phase + '_ik'
                            break
                        check = collision.check(q, opening)
                        if check['collision']:
                            reason = phase + '_collision'
                            break
                        path.append(dict(phase=phase, joints_deg=np.rad2deg(q).tolist(),
                                         flange_xyz_wpr=to_wpr(T), opening_m=opening))
                    if reason:
                        break
            if reason:
                result['rejected'].append(dict(rank=rank, reason=reason, collision=check))
                if reason == 'planning_timeout':
                    result['status'] = reason
                    break
                continue
            result['status'] = 'sampled_path_found'
            result['selected'] = dict(rank=rank, score=float(scores[rank]),
                hand_pose_dataset_m=H.tolist(), jaw_opening_m=opening,
                observed_pad_limits_m=[float(low), float(high)],
                closing_command_m=max(0, min(opening, float(high - low) - .002)),
                home_xyz_wpr=HOME_POSE, pregrasp_xyz_wpr=to_wpr(pre),
                grasp_xyz_wpr=to_wpr(flange), lift_xyz_wpr=to_wpr(lift), path=path)
            break
    result['seconds'] = time.perf_counter() - started
    if result['seconds'] > budget_s:
        result['over_budget_candidate'] = result['selected']
        result['selected'] = None
        result['status'] = 'planning_timeout'
    result['scope'] = 'Sampled geometric planning; native validation and between-sample checks still required'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cloud', type=Path, required=True)
    p.add_argument('--grasps', type=Path, required=True)
    p.add_argument('--condition', choices=['single', 'neighbours', 'bin_walls'], required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = plan(a.cloud, a.grasps, a.condition, a.out)
    print(json.dumps({k: r[k] for k in ['status', 'seconds', 'rejected']}))
