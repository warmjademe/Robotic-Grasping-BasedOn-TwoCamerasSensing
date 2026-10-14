"""Verify that the unused static-fixture filter does not affect reported contact data.

The original SDK logged a count mismatch for the unused static filter. The
evaluation already subtracts target-specific force from total force instead.
Run the same complete 21-case group with only the target filter, and compare.
"""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
import numpy as np

RUN=Path(__file__).resolve().parents[1]
BASE=Path('/home/qyb/RESEARCH/baseline_graspgen')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['nominal','friction_05'],default='nominal')
    profile=parser.parse_args().profile;friction=1. if profile=='nominal' else .5
    source=(RUN/'source_snapshot/run_arm.py').read_text()
    old="filter_prim_paths_expr=['/World/envs/env_.*/Fixtures/.*','/World/envs/env_.*/Target']"
    assert source.count(old)==1
    edited=source.replace(old,"filter_prim_paths_expr=['/World/envs/env_.*/Target']")
    marker="(out/'result.json').write_text(json.dumps(report,indent=2)+'\\n')"
    assert edited.count(marker)==1
    edited=edited.replace(marker,"report['actual_robot_materials_first_env'] = robot.root_physx_view.get_material_properties()[0].cpu().tolist()\n"
        "report['actual_robot_effort_limits_first_env'] = robot.data.joint_effort_limits[0].cpu().tolist()\n"+marker)
    destination=RUN/'diagnostics';destination.mkdir(exist_ok=True)
    variant=destination/'run_arm_target_filter.py';variant.write_text(edited)
    result_dir=RUN/'results/arm/contact_filter_audit'/('mounting_block_01' if profile=='nominal' else 'friction_05/mounting_block_01')
    log=RUN/'logs'/('contact_filter_audit'+('' if profile=='nominal' else '_friction_05')+'.log')
    if not (result_dir/'result.json').exists():
        with log.open('w') as f:
            subprocess.run([str(BASE/'.venv-sim/bin/python'),'-u',str(variant),'--group','mounting_block_01',
                '--output',str(result_dir),'--friction',str(friction)],stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,LD_LIBRARY_PATH=str(BASE/'.sim-libs/lib')),check=True,timeout=600)
    old_dir=RUN/'results/arm'/profile/'mounting_block_01'
    before=json.loads((old_dir/'result.json').read_text());after=json.loads((result_dir/'result.json').read_text())
    fields=['case_id','physics_lift_success','execution_success','collision_free','approach_endpoint_reached','returned_to_origin_success','prohibited_contact_steps']
    assert [{k:r[k] for k in fields} for r in before['rows']]==[{k:r[k] for k in fields} for r in after['rows']]
    a=np.load(old_dir/'trace.npz');b=np.load(result_dir/'trace.npz')
    errors={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['object_pose','bodies','contact_forces','joints']}
    errors['target_contact']=float(np.max(np.abs(a['finger_contact_matrix'][...,-1,:]-b['finger_contact_matrix'][...,-1,:])))
    assert all(v<1e-5 for v in errors.values()),errors
    assert 'Filter pattern' not in log.read_text()
    report=dict(status='passed',profile=profile,cases=21,unchanged_decisions=True,max_absolute_trace_differences=errors,
        actual_robot_materials=after['actual_robot_materials_first_env'],actual_robot_effort_limits=after['actual_robot_effort_limits_first_env'],
        variant_sha256=hashlib.sha256(variant.read_bytes()).hexdigest(),
        conclusion='Removing the unused static-fixture filter eliminates SDK errors and preserves the consumed target forces, trajectories and all case decisions in this complete group.')
    (destination/('contact_filter_audit'+('' if profile=='nominal' else '_friction_05')+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
