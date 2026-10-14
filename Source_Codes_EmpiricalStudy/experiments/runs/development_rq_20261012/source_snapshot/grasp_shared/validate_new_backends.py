#!/usr/bin/env python3
"""CPU interface regression and source-derived geometry audit; no model/robot run.

Synthetic backend outputs test dispatch and serialization only. The frozen real
perception observations are read unchanged. Real checkpoint inference is audited
separately in each method's results, never inferred from these fixtures.
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import xml.etree.ElementTree as ET

import numpy as np

from contract import load, sha, validate

ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geometry_audit():
    base = ROOT / 'baseline_graspgen'
    urdf = base / 'upstream/GraspGen/assets/franka/franka_panda.urdf'
    mesh = urdf.parent / 'meshes/collision/finger.obj'
    robot = ET.parse(urdf).getroot()
    joints = [robot.find("joint[@name='panda_finger_joint%d']" % i) for i in [1, 2]]
    origins = [np.fromstring(j.find('origin').attrib['xyz'], sep=' ') for j in joints]
    axes = [np.fromstring(j.find('axis').attrib['xyz'], sep=' ') for j in joints]
    assert np.array_equal(origins[0], origins[1])
    assert np.array_equal(axes[0], [0, 1, 0]) and np.array_equal(axes[1], [0, -1, 0])
    stroke = sum(float(j.find('limit').attrib['upper']) for j in joints)
    vertices = np.asarray([list(map(float, line.split()[1:4])) for line in mesh.read_text().splitlines() if line.startswith('v ')])
    tip = float(origins[0][2] + vertices[:, 2].max())
    assert np.isclose(stroke, .08) and np.isclose(tip, .112249, atol=1e-9)
    manifest = base / 'fanuc/assets/manifest.json'
    asset = next(x for x in json.loads(manifest.read_text())['assets'] if x['name'] == 'graspgen_finger')
    visual_tip = float(origins[0][2] + asset['bounds_mm'][1][2] / 1000)
    config = base / 'fanuc/config.json'
    cfg = json.loads(config.read_text())
    assert np.isclose(np.asarray(cfg['flange_from_gripper_hand_m'])[2, 3], .04)
    assert np.isclose(.0984 - .04, origins[0][2])
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    assert np.isclose(np.linalg.det(rotation), 1)
    assert np.array_equal(rotation @ [0, 0, 1], [1, 0, 0])
    assert np.array_equal(rotation @ [0, 1, 0], [0, 1, 0])
    offsets = np.array([.01, .02, .03, .04, .06])
    mapped = np.stack([rotation @ [0, 0, tip] + [d-tip, 0, 0] for d in offsets])
    assert np.allclose(mapped, np.column_stack([offsets, offsets*0, offsets*0]), atol=1e-12)
    sources = [urdf, mesh, manifest, config, base/'fanuc/station.py', base/'fanuc/build_assets.py', base/'fanuc/commission.py']
    model_evidence = {}
    for model, paths in {
        'graspnet_baseline': ['upstream/graspnet-baseline/utils/collision_detector.py', 'upstream/graspnetAPI/graspnetAPI/utils/utils.py'],
        'gpd': ['upstream/gpd/src/gpd/candidate/hand.cpp', 'upstream/gpd/include/gpd/candidate/hand.h', 'upstream/gpd/src/gpd/candidate/finger_hand.cpp', 'upstream/gpd/src/gpd/candidate/hand_set.cpp'],
    }.items():
        files = [ROOT / ('baseline_' + model) / p for p in paths]
        sources.extend(p for p in files if p.exists())
        model_evidence[model] = {str(p.relative_to(ROOT)): sha(p) for p in files if p.exists()}
    return dict(
        passed=True, max_width_m=stroke, finger_joint_z_m=float(origins[0][2]),
        collision_finger_zmax_m=float(vertices[:, 2].max()), panda_hand_distal_tip_plane_z_m=tip,
        native_visual_distal_tip_plane_z_m=visual_tip,
        visual_minus_collision_tip_mm=(visual_tip-tip)*1000,
        R_model_common=rotation.tolist(), t_model_common_formula='[author_tip_distance_from_native_origin - panda_hand_tip_z, 0, 0]',
        mapping_direction='T_world_common = T_world_model @ T_model_common; columns express common basis in model coordinates',
        graspnet_origin='Author pose at target seed; finger tips at model x=predicted depth',
        gpd_origin='getPosition is closing-box base center; finger tips at model x=top-bottom=hand_depth',
        gpd_width_note='getGraspWidth is observed closing-region object span; not neural width prediction or candidate aperture',
        alignment_scope='Distal finger-tip plane and closing-axis alignment between modeled geometries; contact surfaces and full volumes are not identical',
        native_capture_band_hand_z_m=[.090, .113],
        actual_robot_tcp_calibrated=False, force_contact_validated=False, native_robot_execution_performed=False,
        checked_tip_offsets_m=offsets.tolist(), max_projection_error_m=float(abs(mapped[:, 0]-offsets).max()),
        sources={str(p.relative_to(ROOT)): sha(p) for p in sources}, model_evidence=model_evidence)


def run_fixture(module, method, input_path, out, payload, environment_map, chdir=False):
    """Run real dispatcher with isolated deterministic backend output fixtures."""
    old_prefix, old_argv, old_cwd = sys.prefix, sys.argv, os.getcwd()
    old_backend = sys.modules.get('backends')
    old_plugin = getattr(module, 'plugin_inference', None)
    old_root = module.ROOT
    arrays, _ = load(input_path)
    called = []
    fake = types.ModuleType('backends')

    def output(label, *args):
        expected = [arrays['target_points_m']]
        if label in ('6dof_graspnet', 'contact_graspnet'):
            expected.append(arrays['T_world_reference'])
        if label == 'contact_graspnet':
            expected.append(arrays['scene_points_m'])
        assert len(args) == len(expected)
        for x, y in zip(args, expected):
            assert np.array_equal(x, y)
        called.append(label)
        if chdir:
            os.chdir(out.parent)
        return copy.deepcopy(payload)

    fake.sixdof = lambda *args: output('6dof_graspnet', *args)
    fake.contact = lambda *args: output('contact_graspnet', *args)
    fake.graspgen = lambda *args: output('graspgen', *args)

    def plugin(label, actual_arrays, metadata, workdir):
        assert workdir.is_absolute()
        for key in arrays:
            assert np.array_equal(actual_arrays[key], arrays[key])
        called.append(label)
        return copy.deepcopy(payload)

    try:
        module.ROOT = ROOT
        sys.prefix = str(ROOT / environment_map[method])
        sys.argv = [module.__file__, '--method', method, '--input', str(input_path), '--out', str(out)]
        sys.modules['backends'] = fake
        if old_plugin is not None:
            module.plugin_inference = plugin
        with contextlib.redirect_stdout(io.StringIO()):
            module.main()
        with np.load(out/'grasps.npz', allow_pickle=False) as data:
            saved = {k: data[k] for k in data.files}
        return saved, json.loads((out/'grasp_generation.json').read_text()), called
    finally:
        sys.prefix, sys.argv = old_prefix, old_argv
        os.chdir(old_cwd)
        module.ROOT = old_root
        if old_backend is None:
            sys.modules.pop('backends', None)
        else:
            sys.modules['backends'] = old_backend
        if old_plugin is not None:
            module.plugin_inference = old_plugin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'validation/new_backend_interface_review.json')
    args = parser.parse_args()
    current = import_file('current_dispatch_review', ROOT/'shared/infer.py')
    previous = import_file('previous_dispatch_review', ROOT/'validation/source_snapshots/before_gpd_graspnet/infer.py')
    methods = ['6dof_graspnet', 'contact_graspnet', 'graspgen', 'gpd', 'graspnet_baseline']
    assert list(current.METHOD_IDS) == methods
    rows = []

    def passed(name, **details):
        rows.append(dict(test=name, passed=True, **details))

    def rejected(name, call):
        try:
            call()
        except (ValueError, KeyError, AssertionError) as e:
            passed(name, rejected_by=type(e).__name__, reason=str(e))
        else:
            raise AssertionError(name + ' was unexpectedly accepted')

    geometry = geometry_audit()
    passed('source_derived_tip_mapping', max_error_m=geometry['max_projection_error_m'])
    actual_adapter_hashes = {}
    gpd_path = ROOT/'baseline_gpd/backend.py'
    if gpd_path.exists():
        gpd = import_file('gpd_geometry_review', gpd_path)
        adapter = gpd.model_to_common()
        current.validate_pose_batch(adapter[None], 1, 'GPD actual adapter')
        cfg_path = ROOT/'baseline_gpd/upstream/gpd/cfg/hand_geometry.cfg'
        config = {line.split('=', 1)[0].strip(): float(line.split('=', 1)[1]) for line in cfg_path.read_text().splitlines() if '=' in line and not line.lstrip().startswith('#')}
        fingertip = adapter @ [0, 0, geometry['panda_hand_distal_tip_plane_z_m'], 1]
        assert np.allclose(fingertip, [config['hand_depth'], 0, 0, 1], atol=1e-12)
        assert np.isclose(gpd.HAND_DEPTH_M, config['hand_depth'])
        actual_adapter_hashes['gpd'] = sha(gpd_path)
        passed('gpd_actual_adapter_matches_author_config_and_panda_mesh', author_base_tip_m=config['hand_depth'])
    graspnet_path = ROOT/'baseline_graspnet_baseline/backend.py'
    if graspnet_path.exists():
        graspnet = import_file('graspnet_geometry_review', graspnet_path)
        for predicted_depths in [np.array([]), np.array([.01, .02, .03, .04])]:
            batch, evidence = graspnet.model_to_common(predicted_depths)
            current.validate_pose_batch(batch, len(predicted_depths), 'GraspNet actual adapters')
            tip_points = batch @ np.array([0, 0, geometry['panda_hand_distal_tip_plane_z_m'], 1])
            expected = np.column_stack([predicted_depths, predicted_depths*0, predicted_depths*0, np.ones(len(predicted_depths))])
            assert np.allclose(tip_points, expected, atol=1e-12)
            assert evidence['physical_tcp_calibrated'] is False
        actual_adapter_hashes['graspnet_baseline'] = sha(graspnet_path)
        passed('graspnet_actual_adapter_all_depth_bins_and_empty')
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, 3] = [[.1, .2, .3], [-.2, .4, .6], [.02, -.03, .5]]
    poses[1, :3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    scores, widths = np.array([.3, .8, -.1]), np.array([.04, .08, .09])
    adapters = np.repeat(np.eye(4)[None], 3, axis=0)
    adapters[:, :3, :3] = geometry['R_model_common']
    depths = np.array([.01, .025, .04])
    adapters[:, 0, 3] = depths - geometry['panda_hand_distal_tip_plane_z_m']
    record = dict(common_grasp_origin='panda_hand', geometry_alignment='Synthetic fixture: verified modeled tip-plane alignment',
                  T_model_to_common_grasp=adapters.tolist(), width_source='network_prediction')
    root_hashes = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'shared/infer.py', ROOT/'shared/backends.py', ROOT/'shared/contract.py', ROOT/'shared/run_from_legacy.py']}
    inputs = ROOT/'validation/inputs_20261011_new_methods'
    input_hashes = {}
    with tempfile.TemporaryDirectory(prefix='grasp-interface-review-') as temp:
        temp = Path(temp)
        for route in ['stereo_v2', 'multiview']:
            path = inputs/(route+'.npz')
            arrays, metadata = load(path)
            input_hashes[route] = dict(npz_sha256=sha(path), json_sha256=sha(path.with_suffix('.json')))
            old_npz = ROOT/'validation/inputs'/(route+'.npz')
            differences = {}
            with np.load(old_npz, allow_pickle=False) as historical:
                for key in historical.files:
                    x, y = historical[key], arrays[key]
                    assert x.shape == y.shape, (route, key)
                    exact = np.array_equal(x, y)
                    delta = float(np.max(np.abs(x-y)))
                    # The full-K re-export changes stereo scene backprojection
                    # by at most float32 rounding; never claim byte equality.
                    if route == 'stereo_v2' and key == 'scene_points_m':
                        assert delta <= 1e-7, (route, key, delta)
                    else:
                        assert exact, (route, key, delta)
                    differences[key] = dict(exact=exact, max_abs_difference=delta)
            passed(route+'_historical_geometry_comparison', fields=differences,
                   full_scene_tolerance_m=1e-7 if route == 'stereo_v2' else 0)
            for method in methods[:3]:
                payload = (poses, scores, widths, dict(method=method))
                old, _, _ = run_fixture(previous, method, path, temp/(route+method+'old'), payload, current.ENVIRONMENTS, chdir=True)
                new, rec, called = run_fixture(current, method, path, temp/(route+method+'new'), payload, current.ENVIRONMENTS, chdir=True)
                assert called == [method]
                for key, value in old.items():
                    assert np.array_equal(new[key], value), (method, key)
                assert rec['native_robot_execution_performed'] is False
                passed(route+'_'+method+'_dispatch_and_old_numeric_outputs_unchanged')
            for method in methods[3:]:
                new, rec, called = run_fixture(current, method, path, temp/(route+method), (poses, scores, widths, record), current.ENVIRONMENTS)
                assert called == [method]
                assert np.array_equal(new['T_world_grasp'], poses @ adapters)
                assert np.array_equal(new['jaw_width_m'], widths)
                assert new['gripper_width_valid'].tolist() == [True, True, False]
                assert rec['input_npz_sha256'] == sha(path)
                assert rec['native_robot_execution_performed'] is False
                passed(route+'_'+method+'_per_candidate_mapping_width_and_fixed_input')
                empty = dict(record, T_model_to_common_grasp=np.empty((0, 4, 4)).tolist())
                output, rec, _ = run_fixture(current, method, path, temp/(route+method+'empty'), (np.empty((0, 4, 4)), [], [], empty), current.ENVIRONMENTS)
                assert rec['status'] == 'no_candidates' and rec['returned'] == 0
                assert output['T_world_grasp'].shape == (0, 4, 4)
                passed(route+'_'+method+'_empty_preserved')

        path = inputs/'stereo_v2.npz'
        for field, value in [('purpose', 'evaluation_only'), ('geometry_source', 'renderer_depth'), ('renderer_depth_used', True), ('object_pose_used', True)]:
            arrays, meta = load(path)
            meta[field] = value
            rejected('reject_input_'+field, lambda: validate(arrays, meta))
        for field, value in [('closing_axis', '+x'), ('approach_axis', '+x'), ('tcp_frame', 'seed_point')]:
            arrays, meta = load(path)
            meta['gripper_geometry'][field] = value
            rejected('reject_input_gripper_'+field, lambda: validate(arrays, meta))
        for label, edited in [('nonhomogeneous', (3, 0, .1)), ('nonrigid', (0, 0, 1.2)), ('nonfinite', (0, 3, float('nan')))]:
            bad = poses.copy()
            i, j, value = edited
            bad[0, i, j] = value
            rejected('reject_author_'+label, lambda: current.validate_pose_batch(bad, 3, 'fixture'))
        for field, value in [('common_grasp_origin', 'seed_point'), ('geometry_alignment', ''), ('width_source', 'unknown')]:
            bad_record = dict(record, **{field: value})
            rejected('reject_plugin_'+field, lambda: run_fixture(current, 'graspnet_baseline', path, temp/('reject_'+field), (poses, scores, widths, bad_record), current.ENVIRONMENTS))
        bad_adapter = adapters.copy()
        bad_adapter[0, 3, 0] = .1
        rejected('reject_nonhomogeneous_adapter', lambda: run_fixture(current, 'graspnet_baseline', path, temp/'bad_adapter', (poses, scores, widths, dict(record, T_model_to_common_grasp=bad_adapter)), current.ENVIRONMENTS))

        # Test actual plugin file selection/loading, without importing either model.
        original_root = current.ROOT
        try:
            current.ROOT = temp/'plugin_fixture'
            for method in methods[3:]:
                folder = current.ROOT/current.PLUGIN_FOLDERS[method]
                folder.mkdir(parents=True)
                backend = folder/'backend.py'
                backend.write_text('def infer(arrays, metadata, workdir):\n    return [], [], [], {"fixture_method": '+repr(method)+'}\n')
                *_, rec = current.plugin_inference(method, {}, {}, temp)
                assert rec['fixture_method'] == method
                assert rec['backend_source_sha256'][str(backend.relative_to(current.ROOT))] == sha(backend)
                passed(method+'_plugin_file_dispatch_and_source_hash')
        finally:
            current.ROOT = original_root

        # Check the real CLI environment selection without executing Python or
        # importing CUDA. Each fake executable only satisfies the path gate.
        old_root, old_argv, old_execv = current.ROOT, sys.argv, os.execv
        env_before = {k: os.environ.get(k) for k in ['LD_LIBRARY_PATH', 'CUDA_LAUNCH_BLOCKING']}
        class CapturedExec(Exception):
            pass
        captured = []
        def capture_exec(executable, argv):
            captured.append((executable, argv))
            raise CapturedExec()
        try:
            current.ROOT = temp/'environment_fixture'
            os.execv = capture_exec
            for method in methods:
                executable = current.ROOT/current.ENVIRONMENTS[method]/'bin/python'
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_text('not executable; intercepted environment selection fixture\n')
                sys.argv = [current.__file__, '--method', method, '--input', str(path), '--out', str(temp/('env_'+method))]
                try:
                    current.main()
                except CapturedExec:
                    pass
                else:
                    raise AssertionError('Expected environment selection before model inference')
                assert captured[-1][0] == str(executable)
                assert captured[-1][1][2:] == sys.argv[1:]
                passed(method+'_isolated_environment_dispatch')
        finally:
            current.ROOT, sys.argv, os.execv = old_root, old_argv, old_execv
            for key, value in env_before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    assert root_hashes == {name: sha(ROOT/name) for name in root_hashes}, 'Reviewed source changed during validation'
    result = dict(passed=True, tests=rows, test_count=len(rows), geometry=geometry, source_sha256=root_hashes,
                  actual_adapter_backend_sha256=actual_adapter_hashes,
                  input_hashes=input_hashes, method_ids=methods, environments=current.ENVIRONMENTS,
                  scope='CPU source audit plus synthetic-backend interface tests on frozen real perception inputs',
                  learned_models_executed=False, existing_gpu_results_unchanged=True, native_robot_execution_performed=False,
                  pending=['Each new backend official checkpoint strict load and real inference evidence', 'Panda volume collision and physical TCP calibration before robot execution'],
                  provenance_boundary='Input declarations and byte hashes are checked; arbitrary geometric origin cannot be authenticated from points alone')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(dict(passed=True, tests=len(rows), out=str(args.out))))


if __name__ == '__main__':
    main()
