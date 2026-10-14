"""Evaluation-only bilateral enclosure for the commissioned parallel gripper.

This is a geometric contact predicate, not a force/friction simulation. Callers
must never use this evaluator's mesh, contact locations, or verdict to select a
candidate or retry a failed grasp. The returned contact opening describes the
simulator response to one already selected closing command.
"""
import numpy as np


def _clip(poly, axis, boundary, keep_above):
    result = []
    if not len(poly):
        return result
    previous = poly[-1]
    previous_inside = (previous[axis] >= boundary) if keep_above else (previous[axis] <= boundary)
    for current in poly:
        inside = (current[axis] >= boundary) if keep_above else (current[axis] <= boundary)
        if inside != previous_inside:
            fraction = (boundary - previous[axis]) / (current[axis] - previous[axis])
            result.append(previous + fraction * (current - previous))
        if inside:
            result.append(current)
        previous, previous_inside = current, inside
    return result


def evaluate(mesh_world, world_from_hand, initial_opening_m, closing_command_m,
             bilateral_gap_tolerance_m=.002, contact_surface_tolerance_m=.0005):
    """Evaluate the target mesh after the commanded grasp pose is reached.

    The first touching finger stops symmetric closure. The opposite pad must
    then lie within 2 mm of a surface with an opposing outward normal. We clip
    triangles to the finite pad region rather than depending on mesh vertices
    happening to fall inside it. Pad dimensions come from the commissioned STL.
    """
    H = np.asarray(world_from_hand, dtype=float)
    result = dict(grip_geometry_valid=False, capture_mode='geometric_bilateral_enclosure',
                  force_contact_simulated=False, initial_opening_m=float(initial_opening_m),
                  closing_command_m=float(closing_command_m),
                  bilateral_gap_tolerance_m=bilateral_gap_tolerance_m,
                  contact_surface_tolerance_m=contact_surface_tolerance_m,
                  pad_region_hand_m=dict(x=[-.010, .010], z=[.090, .113]),
                  failure_reason=None, actual_opening_m=float(closing_command_m))
    if not (0 <= closing_command_m <= initial_opening_m <= .08):
        result['failure_reason'] = 'invalid_closure_command'
        return result
    triangles = (mesh_world.triangles - H[:3, 3]) @ H[:3, :3]
    normals = mesh_world.face_normals @ H[:3, :3]
    points, point_normals = [], []
    for triangle, normal in zip(triangles, normals):
        poly = list(triangle)
        for axis, bound, above in [(0, -.010, True), (0, .010, False),
                                   (2, .090, True), (2, .113, False)]:
            poly = _clip(poly, axis, bound, above)
        if len(poly) >= 3:
            points.extend(poly)
            point_normals.extend([normal] * len(poly))
    if not points:
        result['failure_reason'] = 'no_target_surface_between_pads'
        return result
    points, point_normals = np.asarray(points), np.asarray(point_normals)
    low, high = points[:, 1].min(), points[:, 1].max()
    result['surface_limits_hand_y_m'] = [float(low), float(high)]
    if not low < 0 < high:
        result['failure_reason'] = 'target_not_between_fingers'
        return result
    contact_opening = 2 * max(-low, high)
    opposite_gap = abs(high + low)
    result.update(first_contact_opening_m=float(contact_opening),
                  opposite_pad_gap_m=float(opposite_gap))
    if initial_opening_m + contact_surface_tolerance_m < contact_opening:
        result['failure_reason'] = 'target_intersects_initial_finger_opening'
        return result
    if closing_command_m > contact_opening + contact_surface_tolerance_m:
        result['failure_reason'] = 'closure_stops_before_contact'
        return result
    result['actual_opening_m'] = float(contact_opening)
    if opposite_gap > bilateral_gap_tolerance_m:
        result['failure_reason'] = 'one_sided_contact'
        return result
    left = points[:, 1] >= high - contact_surface_tolerance_m
    right = points[:, 1] <= low + contact_surface_tolerance_m
    threshold = np.cos(np.deg2rad(45))
    left_ok = bool(np.any(point_normals[left, 1] >= threshold))
    right_ok = bool(np.any(point_normals[right, 1] <= -threshold))
    result['opposing_surface_normals'] = dict(left=left_ok, right=right_ok,
                                            max_angle_deg=45)
    if not (left_ok and right_ok):
        result['failure_reason'] = 'no_opposing_contact_normals'
        return result
    result['grip_geometry_valid'] = True
    return result


def calibration_cases():
    """Independent synthetic geometry cases; never part of the RGB benchmark."""
    import trimesh
    box = trimesh.creation.box([.030, .030, .020])
    box.apply_translation([0, 0, .1015])
    cases = []
    for name, offset, initial, command, expected in [
            ('centered_closed', 0., .08, .02, True),
            ('centered_left_open', 0., .08, .08, False),
            ('off_target', .06, .08, .02, False),
            ('one_sided', .004, .08, .02, False),
            ('initial_penetration', 0., .02, .01, False)]:
        H = np.eye(4)
        H[1, 3] = offset
        result = evaluate(box, H, initial, command)
        assert result['grip_geometry_valid'] is expected, (name, result)
        cases.append(dict(case=name, expected=expected, result=result))
    return cases


if __name__ == '__main__':
    import json
    print(json.dumps(calibration_cases(), indent=2))
