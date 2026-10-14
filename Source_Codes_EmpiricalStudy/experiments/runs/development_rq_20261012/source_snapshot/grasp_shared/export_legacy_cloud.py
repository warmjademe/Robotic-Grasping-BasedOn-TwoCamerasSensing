"""Bridge saved stereo or multiview observations into the common input contract.

Only measurements and calibration are read. No scene pose or renderer depth is
loaded; geometry evaluation and robot execution remain outside this adapter.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from contract import save, sha, transform_points


def prediction_record(source):
    """Require provenance from the audited legacy prediction format."""
    path = Path(source)/'reconstruction.json'
    record = json.loads(path.read_text())
    if record.get('purpose') in ['evaluation_only', 'ground_truth'] or record.get('geometry_source') in ['ground_truth', 'renderer_depth']:
        raise ValueError('Evaluation/GT records cannot enter grasp inference')
    prohibited = ['renderer_depth_used', 'renderer_depth_used_for_reconstruction',
                  'object_pose_used', 'object_pose_used_for_reconstruction', 'geometry_truth_used']
    if any(record.get(k) is True for k in prohibited):
        raise ValueError('Source record declares GT geometry')
    depth_declared = (record.get('geometry_truth_used') is False or
                      record.get('renderer_depth_used_for_reconstruction') is False)
    pose_declared = (record.get('object_pose_used') is False or
                     record.get('object_pose_used_for_reconstruction') is False)
    if not depth_declared or not pose_declared or not record.get('method'):
        raise ValueError('Legacy source lacks explicit prediction provenance declarations')
    if record.get('reconstruction_sha256') != sha(Path(source)/'reconstruction.npz'):
        raise ValueError('Source prediction hash does not match reconstruction.json')
    return record, sha(path)


def export(source, out, cameras=None, reference=None):
    source = Path(source)
    provenance, record_hash = prediction_record(source)
    p = source / 'reconstruction.npz'
    with np.load(p, allow_pickle=False) as z:
        cloud = {k: z[k] for k in z.files}
    target = np.asarray(cloud['points'], dtype=np.float32)
    hashes = {'reconstruction.npz': sha(p), 'reconstruction.json': record_hash}
    if cameras is None:
        calibration = source / 'calibration.json'
        c = json.loads(calibration.read_text())
        T = np.asarray(c['world_from_left_camera'], dtype=np.float64)
        origins = [T[:3, 3]]
        if 'world_from_right_camera' in c:
            origins.append(np.asarray(c['world_from_right_camera'])[:3, 3])
        elif 'baseline_m' in c and 'Parallel calibrated pinhole' in c.get('calibration', ''):
            origins.append(T[:3, 3] + T[:3, 0] * float(c['baseline_m']))
        else:
            raise ValueError('The second camera pose is missing and no calibrated parallel baseline is declared')
        hashes['calibration.json'] = sha(calibration)
        route = 'two_cameras_stereo'
    else:
        cameras = Path(cameras)
        cs = json.loads(cameras.read_text())
        cs = cs.get('cameras', cs)
        if reference not in cs:
            raise ValueError('Select an explicit reference from the calibrated cameras')
        T = np.asarray(cs[reference].get('T_world_camera', cs[reference].get('world_from_camera')), dtype=np.float64)
        origins = [np.asarray(v.get('T_world_camera', v.get('world_from_camera')))[:3, 3] for v in cs.values()]
        hashes['cameras.json'] = sha(cameras)
        route = 'multi_camera_reconstruction'
    if all(k in cloud for k in ['depth', 'K', 'valid']) and cameras is None:
        depth, K, valid = cloud['depth'], cloud['K'], cloud['valid'].astype(bool)
        yy, xx = np.indices(depth.shape)
        z = depth[valid]
        pixels = np.column_stack((xx[valid], yy[valid], np.ones(len(z))))
        rays = pixels @ np.linalg.inv(K).T
        scene_camera = (rays/rays[:, 2:3]*z[:, None]).astype(np.float32)
        scene = transform_points(scene_camera, T)
        scope = 'full_observed_scene'
        point_ids = np.zeros(len(scene), dtype=np.int32)
    elif 'scene_points' in cloud:
        scene = np.asarray(cloud['scene_points'])
        scope = 'full_observed_scene'
        point_ids = np.full(len(scene), -1, dtype=np.int32)
    else:
        # Existing multiview runs export a segmented target cloud only. Keep
        # that limitation visible instead of claiming complete collision input.
        scene = target.copy()
        scope = 'target_only'
        point_ids = np.full(len(scene), -1, dtype=np.int32)
    arrays = dict(scene_points_m=scene, target_points_m=target,
                  T_world_reference=T, camera_origins_m=np.asarray(origins),
                  point_camera_ids=point_ids)
    metadata = dict(schema_version=1, units='m', frame_id='experiment_world',
                    purpose='grasp_inference', geometry_source='perception_prediction',
                    reference_convention='opencv_optical_x_right_y_down_z_forward',
                    reference_camera=reference if cameras else 'legacy_left',
                    perception_route=route, scene_scope=scope,
                    gripper_id='franka_panda', gripper_geometry=dict(max_width_m=.08,
                    closing_axis='+y', approach_axis='+z', tcp_frame='panda_hand'),
                    input_hashes=hashes, has_normals=False, has_colors=False,
                    renderer_depth_used=False, object_pose_used=False,
                    source_prediction_method=provenance['method'],
                    source_provenance_check='Legacy reconstruction.json declarations and reconstruction SHA256; not automatic origin authentication',
                    point_source_note='-1 denotes fused or unavailable point-to-camera association')
    save(out, arrays, metadata)
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cameras', type=Path)
    p.add_argument('--reference')
    a = p.parse_args()
    print(json.dumps(export(a.source, a.out, a.cameras, a.reference), ensure_ascii=False))
