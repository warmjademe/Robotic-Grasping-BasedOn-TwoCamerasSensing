# Reproduction

The root README contains the CPU-only table reproduction command. This is the shortest path to checking the reported results. It uses all 72 scenes and both friction profiles, checks missing planned executions, and keeps the shared configuration from being counted twice.

## Source and dependency reconstruction

Run `python3 tools/fetch_upstreams.py` to obtain the recorded upstream commits. Local patches are applied after checkout. Source repositories and their original license files are listed in `tools/upstreams.json`. The compatible TensorFlow inference source used on the execution host is included under `Source_Codes_Grasping_Modules/baseline_runtime/compat`; compiled CUDA operators must be rebuilt for the target host.

The independent environments expected by `shared/infer.py` are `.venv-tf` for 6-DOF GraspNet and Contact-GraspNet, `.venv-infer` for GraspGen, and the respective `.venv` environments for GPD and GraspNet-Baseline. Do not combine their TensorFlow/PyTorch/CUDA requirements into one environment. See the retained `build_tf_ops.py`, `prepare_6dof_compat.py`, `build_ops.sh`, `build.sh`, build manifests and environment records. Frozen scripts retain the original execution-host paths; configure runtime paths for a new host before launching them. Model/checkpoint hashes are recorded per inference in `perception.json` and `grasp_generation.json`.

No new GPU inference or simulator execution was performed for this publication. The saved traces and actual execution sources are published; recreating the full runtime remains dependent on compatible simulator installations and author weights.

## Recheck recorded point clouds and FANUC physical trajectories

After downloading the release archives:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/pip install -r requirements-analysis.txt
.venv-analysis/bin/python Source_Codes_EmpiricalStudy/RQ1/code/analyze.py --label independent_check
```

This writes a new label and verifies source hashes. Ground-truth geometry is used for evaluation, not supplied to the grasp generator. Fixed scene cropping is part of the perception protocol.

`physics_validation_20261013/code/analyze_results.py` recomputes held-object clearance, contact checks and approach completion from the saved trajectories. `platform_comparison_20261014/code/summarize.py` checks and exports Franka records. Both historical analyzers write their usual output locations: use a disposable copy to keep original result hashes unchanged. `tools/verify_artifacts.py` can detect changed original files.

## Physical criteria

An executed grasp succeeds when the object bottom remains at least 100 mm above the support surface for 2 seconds, approach endpoint error is within 2 mm and 0.5 degrees, and prohibited contacts do not exceed the recorded threshold. The original protocols and per-step contact traces define the exact interpretation. Object pose is not reassigned after initialization. The gripper-only GraspDataGen tug check is a separate diagnostic and is not substituted for full-arm success.

## Manuscript analysis

The full manuscript is `Source_Codes_EmpiricalStudy/latex/main.tex`. The outline is retained separately. Generate the three data-derived figures, candidate/plan tables and exploratory workpiece-paired comparisons with the pinned analysis environment:

```bash
.venv-analysis/bin/python Source_Codes_EmpiricalStudy/publication/paper_analysis.py --out manuscript_reanalysis --figures reproduced_figures
```

The script resamples eight workpieces while retaining their nine paired scenes, uses 20,000 bootstrap replicates with seed 20261014, enumerates 256 sign flips, and applies BH correction separately to the 18 perception and 12 grasp-model comparisons. These analyses reuse the recorded experiment outcomes; they do not run additional robot trials.

## Provenance

`tools/artifacts.json` binds released files to their bytes. The experiment manifests bind inputs, candidates, selected plans, meshes and historical source snapshots. Original timestamps, paths, seeds and failure outcomes are retained. Source snapshots may contain obsolete launchers as historical evidence; current experiment launch control remains disabled. Publishing the repository does not resume an experiment or connect to an industrial robot.
