#!/usr/bin/env python3
"""Generate manuscript figures and object-paired descriptive statistics."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reproduce_tables import reproduce, read_csv, save_csv, STUDY

STEREO=['stereobm','sgm','psmnet','gwcnet','raft_stereo','igev_stereo','foundationstereo']
GRASP=['gpd','6dof_graspnet','contact_graspnet','graspnet_baseline','graspgen']
NAMES={'stereobm':'StereoBM','sgm':'StereoSGBM','psmnet':'PSMNet','gwcnet':'GwcNet',
       'raft_stereo':'RAFT-Stereo','igev_stereo':'IGEV-Stereo','foundationstereo':'FoundationStereo',
       'gpd':'GPD','6dof_graspnet':'6-DOF GraspNet','contact_graspnet':'Contact-GraspNet',
       'graspnet_baseline':'GraspNet-Baseline','graspgen':'GraspGen'}
PLATFORMS=['fanuc_isaac','franka_isaac','franka_mujoco']
LABELS=['FANUC + Isaac Sim','Franka + Isaac Sim','Franka + MuJoCo']
CONDITIONS=['single','neighbours','bin_walls']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--figures',type=Path,required=True)
    args=parser.parse_args()
    reproduce(args.out)
    args.figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12.5,'pdf.fonttype':42})
    summary=read_csv(args.out/'rq2_rq3_success.csv')
    for rq,methods in [('RQ2',STEREO),('RQ3',GRASP)]:
        fig,axes=plt.subplots(1,3,figsize=(11.2,4.2 if rq=='RQ2' else 3.6),sharey=True)
        for ax,platform,label in zip(axes,PLATFORMS,LABELS):
            matrix=np.array([[int(next(r['successes'] for r in summary if r['rq']==rq and
                r['platform']==platform and r['profile']=='nominal' and r['method']==m and r['condition']==c))
                for c in CONDITIONS] for m in methods])
            im=ax.imshow(matrix,cmap='viridis',vmin=0,vmax=24,aspect='auto')
            ax.set_xticks(range(3),['Single','Neighbours','Bin\nwalls'])
            ax.set_yticks(range(len(methods)),[NAMES[m] for m in methods])
            ax.set_title(label)
            for i,j in itertools.product(range(len(methods)),range(3)):
                ax.text(j,i,f'{matrix[i,j]}/24',ha='center',va='center',color='black' if matrix[i,j]>14 else 'white')
        fig.subplots_adjust(left=.18,right=.93,bottom=.15,top=.88,wspace=.12)
        cax=fig.add_axes([.945,.18,.012,.66]);fig.colorbar(im,cax=cax,label='Successful scenes')
        fig.savefig(args.figures/(rq.lower()+'_conditions.pdf'),bbox_inches='tight');plt.close(fig)
    points=read_csv(STUDY/'RQ1/data/development_pointclouds_20261012_v2/per_scene.csv')
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.2))
    for ax,key,label,scale in zip(axes,['surface_mae_mm','precision_2mm','coverage_2mm'],
        ['Mean surface distance (mm)','Points within 2 mm (%)','Visible surface covered (%)'],[1,100,100]):
        arrays=[[float(r[key])*scale for r in points if r['method']==m] for m in STEREO]
        ax.boxplot(arrays,vert=False,tick_labels=[NAMES[m] for m in STEREO],showfliers=True,
                   flierprops={'markersize':2},medianprops={'color':'#d95f02'})
        ax.set_xlabel(label);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        if key=='surface_mae_mm':ax.set_xscale('log')
    for ax in axes[1:]:ax.set_yticklabels([])
    fig.tight_layout();fig.savefig(args.figures/'rq1_distributions.pdf',bbox_inches='tight');plt.close(fig)
    outcomes=read_csv(args.out/'all_task_outcomes.csv')
    # Resample workpieces, keeping all poses/conditions paired. Inference is exploratory.
    objects=sorted({r['episode_id'].rsplit('_p',1)[0] for r in outcomes})
    assert len(objects)==8
    rng=np.random.default_rng(20261014)
    resamples=rng.integers(0,len(objects),size=(20000,len(objects)))
    signs=np.array(list(itertools.product([-1,1],repeat=len(objects))))
    comparisons=[]
    for rq,methods,reference,field,fixed,value in [
        ('RQ2',STEREO,'foundationstereo','perception','grasp','graspgen'),
        ('RQ3',GRASP,'graspgen','grasp','perception','foundationstereo')]:
        for platform in PLATFORMS:
            rates={}
            for method in methods:
                rows=[r for r in outcomes if r['platform']==platform and r['profile']=='nominal'
                      and r[fixed]==value and r[field]==method]
                assert len(rows)==72
                rates[method]=np.array([np.mean([r['success']=='True' for r in rows
                    if r['episode_id'].rsplit('_p',1)[0]==obj]) for obj in objects])
            for method in methods:
                if method==reference:continue
                d=rates[reference]-rates[method]
                lo,hi=np.quantile(d[resamples].mean(axis=1),[.025,.975])
                p=float(np.mean(np.abs((signs*d).mean(axis=1))>=abs(d.mean())-1e-12))
                comparisons.append(dict(rq=rq,platform=platform,reference=reference,comparison=method,
                    difference_pp=float(d.mean()*100),ci_low_pp=float(lo*100),ci_high_pp=float(hi*100),
                    exact_sign_flip_p=p,workpieces=8,bootstrap_samples=20000,seed=20261014))
    for rq in ['RQ2','RQ3']:
        rows=sorted([r for r in comparisons if r['rq']==rq],key=lambda r:r['exact_sign_flip_p'])
        last=1.
        for i in range(len(rows)-1,-1,-1):
            last=min(last,rows[i]['exact_sign_flip_p']*len(rows)/(i+1))
            rows[i]['bh_fdr_p']=last
    save_csv(args.out/'paired_comparisons.csv',comparisons)
    stages=[]
    for rq,folder,field in [('RQ2','development_graspgen_20261012','perception'),('RQ3','development_foundationstereo_20261012','grasp')]:
        rows=read_csv(STUDY/rq/'data'/folder/'per_scene.csv')
        for method in (STEREO if rq=='RQ2' else GRASP):
            group=[r for r in rows if r[field]==method]
            fr=next(r for r in summary if r['rq']==rq and r['method']==method and r['platform']=='franka_isaac' and r['profile']=='nominal' and r['condition']=='all')
            stages.append(dict(rq=rq,method=method,tasks=72,nonempty_candidates=sum(int(r['returned_candidates'])>0 for r in group),
                fanuc_plans=sum(r['path_found']=='True' for r in group),franka_plans=int(fr['selected_plans'])))
    save_csv(args.out/'candidate_plan_stages.csv',stages)
    failure=[]
    fanuc=read_csv(STUDY/'physics_validation_20261013/analysis/per_candidate.csv')
    franka=read_csv(STUDY/'platform_comparison_20261014/analysis/per_task.csv')
    for platform in PLATFORMS:
        rows=[r for r in (fanuc if platform=='fanuc_isaac' else franka) if r['profile']=='nominal'
              and (platform=='fanuc_isaac' or (r['platform']==platform and r['physics_status']=='completed'))]
        failure.append(dict(platform=platform,executed=len(rows),endpoint_failed=sum(r['approach_endpoint_reached']=='False' for r in rows),
            lift_failed=sum(r['physics_lift_success']=='False' for r in rows),contact_failed=sum(r['collision_free']=='False' for r in rows),
            success=sum(r['execution_success']=='True' for r in rows)))
    save_csv(args.out/'physical_failures_overlapping.csv',failure)
    hashes={str(p.relative_to(STUDY)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        STUDY/'RQ1/data/development_pointclouds_20261012_v2/per_scene.csv',
        STUDY/'physics_validation_20261013/analysis/per_candidate.csv',
        STUDY/'platform_comparison_20261014/analysis/per_task.csv']}
    (args.out/'paper_analysis_manifest.json').write_text(json.dumps(dict(input_sha256=hashes,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistics_scope='Exploratory workpiece-paired comparisons; 18 RQ2 and 12 RQ3 tests, BH within each question',
        bootstrap_seed=20261014,bootstrap_samples=20000),indent=2)+'\n')
    print('Generated three figures, paired contrasts, stages and overlapping failure counts')


if __name__=='__main__':main()
