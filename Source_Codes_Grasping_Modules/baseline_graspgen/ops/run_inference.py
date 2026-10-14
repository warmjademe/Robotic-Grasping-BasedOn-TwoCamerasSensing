#!/usr/bin/env python3
"""Auditable headless inference using unmodified official GraspGenSampler."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/GraspGen'))
# These pinned public NVIDIA checkpoints contain serialized config objects.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import torch
import trimesh
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.robot import load_control_points_for_visualization

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def validate(poses, scores, expected):
    assert poses.shape == (expected, 4, 4), poses.shape
    assert scores.shape == (expected,), scores.shape
    assert np.isfinite(poses).all() and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    r = poses[:, :3, :3]
    orthogonality = np.max(np.abs(r.transpose(0, 2, 1) @ r - np.eye(3)))
    determinant = np.linalg.det(r)
    assert orthogonality < 1e-4, orthogonality
    assert np.allclose(determinant, 1, atol=1e-4), determinant
    assert np.allclose(poses[:, 3, :], [0, 0, 0, 1])
    assert np.all(np.diff(scores) <= 1e-6)
    return dict(max_orthogonality_error=float(orthogonality),
                determinant_min=float(determinant.min()),
                determinant_max=float(determinant.max()))

def visualize(output, pc, colors, poses, scores, gripper):
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    controls = np.asarray(load_control_points_for_visualization(gripper)).reshape(-1, 3)
    colors = np.asarray(colors)[:, :3]
    if colors.max() <= 1:
        colors = colors * 255
    colors = colors.clip(0, 255).astype(np.uint8)
    traces = [go.Scatter3d(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers',
        marker=dict(size=2, color=['rgb(%d,%d,%d)' % tuple(c) for c in colors]), name='Input point cloud')]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*pc.T, c=colors / 255, s=3)
    for i, (pose, score) in enumerate(zip(poses[:10], scores[:10])):
        pts = controls @ pose[:3, :3].T + pose[:3, 3]
        traces.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode='lines',
            line=dict(width=5), name=f'Rank {i+1}, score {score:.3f}',
            visible=True if i < 3 else 'legendonly'))
        if i < 3:
            ax.plot(*pts.T, lw=2, label=f'#{i+1}: {score:.3f}')
    mid = pc.mean(axis=0)
    half = max(np.ptp(pc, axis=0).max() * .65, .09)
    for dim, setlim in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setlim(mid[dim] - half, mid[dim] + half)
    ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title=f'{output.name}: predicted grasps')
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(output.with_suffix('.png'), dpi=150)
    plt.close(fig)
    interactive = go.Figure(traces)
    interactive.update_layout(title=f'{output.name} | {gripper} | scores are model outputs',
        scene=dict(aspectmode='data', xaxis_title='x (m)', yaxis_title='y (m)', zaxis_title='z (m)'))
    interactive.write_html(output.with_suffix('.html'), include_plotlyjs=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['real_pc', 'meshes'], required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'A CUDA GPU is required'
    torch.set_num_threads(8)
    gripper = 'robotiq_2f_140' if args.suite == 'real_pc' else 'franka_panda'
    cfg_path = ROOT / f'models/checkpoints/graspgen_{gripper}.yml'
    cfg = load_grasp_cfg(str(cfg_path))
    started = time.perf_counter()
    sampler = GraspGenSampler(cfg)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - started
    assert next(sampler.model.parameters()).is_cuda
    if args.suite == 'real_pc':
        sources = sorted((ROOT / 'models/sample_data/real_object_pc').glob('*.json'))
        topk = 100
    else:
        sources = [ROOT / 'upstream/GraspDataGen/objects/mug.obj',
                   ROOT / 'upstream/GraspDataGen/objects/banana.obj',
                   ROOT / 'models/sample_data/meshes/box.obj']
        topk = 20
    if args.limit:
        sources = sources[:args.limit]
    out = ROOT / 'results' / f'inference_{args.suite}'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    warmed = False
    for source in sources:
        for seed in args.seeds:
            seed_all(seed)
            if args.suite == 'real_pc':
                data = json.loads(source.read_text())
                pc = np.asarray(data['pc'], dtype=np.float32)
                colors = np.asarray(data['pc_color'])
                remove_outliers = True
            else:
                mesh = trimesh.load(source, force='mesh')
                object_scale = .5 if source.name == 'box.obj' else 1.0
                mesh.apply_scale(object_scale)
                pc, _ = trimesh.sample.sample_surface(mesh, 1024, seed=seed)
                pc = np.asarray(pc, dtype=np.float32)
                colors = np.tile([150, 180, 210], (len(pc), 1))
                remove_outliers = False
            if not warmed:
                GraspGenSampler.run_inference(pc, sampler, grasp_threshold=-1,
                    num_grasps=200, topk_num_grasps=topk, min_grasps=topk,
                    max_tries=1, remove_outliers=remove_outliers)
                torch.cuda.synchronize()
                warmed = True
                seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            poses_t, scores_t = GraspGenSampler.run_inference(pc, sampler,
                grasp_threshold=-1, num_grasps=200, topk_num_grasps=topk,
                min_grasps=topk, max_tries=1, remove_outliers=remove_outliers)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            poses = poses_t.cpu().numpy()
            scores = scores_t.cpu().numpy()
            checks = validate(poses, scores, topk)
            target = out / f'{source.stem}_seed{seed}'
            record = dict(input=str(source.relative_to(ROOT)), input_sha256=digest(source),
                object_scale=object_scale if args.suite == 'meshes' else None,
                seed=seed, gripper=gripper, backbone=cfg.diffusion.obs_backbone,
                input_points=len(pc), input_extents_m=np.ptp(pc, axis=0).tolist(),
                num_candidates=200, returned=len(scores), remove_outliers=remove_outliers,
                pose_frame='same as input point cloud; gripper palm origin, +Z approach, X closing',
                unit='metre', latency_seconds=elapsed,
                max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                score_min=float(scores.min()), score_max=float(scores.max()),
                pose_checks=checks, grasp_poses=poses.tolist(), model_scores=scores.tolist())
            np.savez_compressed(target.with_suffix('.npz'), points=pc, colors=colors,
                                grasp_poses=poses, model_scores=scores)
            target.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
            if seed == args.seeds[0]:
                visualize(target, pc, colors, poses, scores, gripper)
            records.append({k: v for k, v in record.items() if k not in ['grasp_poses', 'model_scores']})
            print(json.dumps(records[-1]), flush=True)
    summary = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        suite=args.suite, model_load_seconds=model_load_seconds,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        source_revision='2dd8852e1be60f5f9d277fafcc621835cdf59110',
        model_revision='ec1ccbb5eec0680db669246ac312a3636f16ee43',
        planner='diffusion-discriminator only', deterministic_algorithms=False,
        physical_success_measured=False, runs=records)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    main()
