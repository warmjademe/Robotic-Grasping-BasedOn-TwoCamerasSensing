"""Explicit synthetic unit fixtures; never stored as experimental execution results."""
from copy import deepcopy
from evaluate_episodes import score

episodes=[dict(episode_id='fixture_a',object_id='a',family='family_a',split='test'),dict(episode_id='fixture_b',object_id='b',family='family_b',split='test')]
ok=dict(episode_id='fixture_a',method_id='fixture_model',seed=0,success_mode='geometric',status='completed',
        grasp_success=True,native_execution_performed=True,human_interventions=0,prohibited_collision=False,
        joint_limit_violation=False,target_identity_correct=True,grip_geometry_valid=True,
        bottom_clearance_m=.11,hold_sim_s=2.1,failure_reason=None,evidence_files={'synthetic_fixture_only.txt':'placeholder'},
        execution_sim_s=4.0,task_wall_s=6.0)
bad=dict(ok,episode_id='fixture_b',grasp_success=False,native_execution_performed=False,failure_reason='no_candidates',execution_sim_s=None,task_wall_s=1.)
args=('fixture_model',0,'test','geometric')
assert score(episodes,[],*args)['official_success_rate'] is None
assert score(episodes,[ok],*args)['official_success_rate'] is None
r=score(episodes,[ok,bad],*args)
assert r['official_success_rate']==.5 and r['timing']['execution_sim_s']['n']==1
assert r['failure_counts']['no_candidates']==1 and r['timing']['task_wall_s']['n']==2
tests=4
for mutate in [lambda x:x.update(bottom_clearance_m=.01),lambda x:x.update(hold_sim_s=1),
               lambda x:x.update(prohibited_collision=True),lambda x:x.update(native_execution_performed=False),
               lambda x:x.update(human_interventions=1),lambda x:x.update(success_mode='contact_physics')]:
    value=deepcopy(ok);mutate(value)
    try:score(episodes,[value,bad],*args)
    except AssertionError:tests+=1
    else:raise AssertionError('Invalid success record accepted')
try:score(episodes,[ok,ok],*args)
except AssertionError:tests+=1
else:raise AssertionError('Duplicate successful trials accepted')
print({'passed':True,'checks':tests,'scope':'synthetic evaluator unit tests, not robot trials'})
