"""Official FANUC link geometry and the commissioned Panda tool, in metres.

This module accepts known fixtures or perceived geometry from its caller. It
does not read object identities, true poses, masks, or evaluation directories.
"""
from pathlib import Path
import sys
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
GRASPGEN = PROJECT / 'Source_Codes_Grasping_Modules/baseline_graspgen'
sys.path.insert(0, str(GRASPGEN / 'fanuc'))
from kinematics import ORIGINS, AXES, fk, translation

DATASET_TO_ROBOT = translation([.965, 0, 1.240])
FLANGE_TO_HAND = translation([0, 0, .040])


def fixtures(condition):
    """Fixed installation dimensions, shared across all perception methods."""
    result = [('table', [.65, .5, .035], [0, 0, -.0175])]
    if condition == 'bin_walls':
        result += [('bin_wall_0', [.32, .012, .07], [0, -.105, .035]),
                   ('bin_wall_1', [.32, .012, .07], [0, .105, .035]),
                   ('bin_wall_2', [.012, .21, .07], [-.154, 0, .035]),
                   ('bin_wall_3', [.012, .21, .07], [.154, 0, .035])]
    elif condition not in ['single', 'neighbours']:
        raise ValueError('Unknown installation condition')
    return result


def link_frames(q):
    q = np.asarray(q, dtype=float).copy()
    q[2] += q[1]  # FANUC controller J3 is coupled to J2.
    T = np.eye(4)
    result = {'base': T.copy()}
    for i, (origin, axis, angle) in enumerate(zip(ORIGINS, AXES, q), 1):
        R = np.eye(4)
        R[:3, :3] = Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix()
        T = T @ translation(origin) @ R
        result['j' + str(i)] = T.copy()
    return result


class RobotCollision:
    def __init__(self, condition, scene_points=None, voxel_m=.003):
        self.robot = trimesh.collision.CollisionManager()
        self.environment = trimesh.collision.CollisionManager()
        self.names = ['base'] + ['j' + str(i) for i in range(1, 7)]
        asset = HERE / 'robot_assets/fanuc_description/fanuc_m20_description/meshes/m20_25_18d/collision'
        self.meshes = {name: trimesh.load(asset / (name + '.stl'), force='mesh') for name in self.names}
        for name, mesh in self.meshes.items():
            self.robot.add_object(name, mesh)
        palm = trimesh.load(GRASPGEN / 'fanuc/assets/graspgen_palm.stl', force='mesh')
        finger = trimesh.load(GRASPGEN / 'fanuc/assets/graspgen_finger.stl', force='mesh')
        palm.apply_scale(.001)
        finger.apply_scale(.001)
        self.meshes.update(palm=palm, finger_left=finger, finger_right=finger)
        for name in ['palm', 'finger_left', 'finger_right']:
            self.robot.add_object(name, self.meshes[name])
        # Adjacent surfaces intentionally share a mechanical joint or mount.
        self.allowed_pairs = {frozenset(pair) for pair in zip(self.names, self.names[1:])}
        self.allowed_pairs.update(frozenset(pair) for pair in [
            # J5 and J6 have coincident joint origins in the official URDF;
            # the J4 housing and J6 shaft meshes overlap inside this wrist.
            ('j4', 'j6'), ('j6', 'palm'),
            ('palm', 'finger_left'), ('palm', 'finger_right')])
        for name, extents, center in fixtures(condition):
            self.environment.add_object(name, trimesh.creation.box(extents),
                                        DATASET_TO_ROBOT @ translation(center))
        self.voxel_count = 0
        if scene_points is not None:
            points = np.asarray(scene_points)
            keep = (points[:, 2] > .003) & (points[:, 2] < .30)
            keep &= (np.abs(points[:, 0]) < .325) & (np.abs(points[:, 1]) < .25)
            points = points[keep]
            if len(points):
                cells = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0)
                centers = (cells + .5) * voxel_m
                self.voxel_count = len(centers)
                # Occupancy is conservative within each observed voxel. Unknown
                # occluded geometry remains unknown to the planner.
                mesh = trimesh.voxel.ops.multibox(centers, pitch=voxel_m)
                self.environment.add_object('perceived_scene', mesh, DATASET_TO_ROBOT)

    def pose(self, joints_rad, opening_m):
        if not np.isfinite(opening_m) or not 0 <= opening_m <= .08:
            raise ValueError('Predicted gripper opening is outside 0..80 mm')
        frames = link_frames(joints_rad)
        flange = fk(joints_rad)
        frames['palm'] = flange
        for name, sign in [('finger_left', 1), ('finger_right', -1)]:
            T = translation([0, sign * opening_m / 2, .0984])
            if sign < 0:
                T[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix()
            frames[name] = flange @ T
        for name, T in frames.items():
            self.robot.set_transform(name, T)
        return frames

    def check(self, joints_rad, opening_m):
        self.pose(joints_rad, opening_m)
        _, internal = self.robot.in_collision_internal(return_names=True)
        internal = [sorted(pair) for pair in internal if frozenset(pair) not in self.allowed_pairs]
        _, external = self.robot.in_collision_other(self.environment, return_names=True)
        return {'collision': bool(internal or external), 'self_pairs': sorted(internal),
                'environment_pairs': sorted([list(p) for p in external]),
                'perceived_voxels': self.voxel_count}
