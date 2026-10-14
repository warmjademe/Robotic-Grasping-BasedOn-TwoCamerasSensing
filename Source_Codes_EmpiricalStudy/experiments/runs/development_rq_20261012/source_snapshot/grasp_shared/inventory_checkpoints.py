"""Fingerprint actual checkpoint bytes without loading models or running GPUs."""
import json
from pathlib import Path
import re
from datetime import datetime,timezone
from contract import sha


def main():
    root=Path(__file__).resolve().parents[1]
    groups={}
    for model in ['ACRONYM_VAE','ACRONYM_Evaluator']:
        folder=root/'baseline_6dof_graspnet/models'/model
        tfout=folder/'tf_output'
        state=(tfout/'checkpoint').read_text()
        prefix=tfout/Path(re.search(r'model_checkpoint_path:\s*"([^"]+)"',state).group(1)).name
        files=list(tfout.glob(prefix.name+'.*'))
        assert any('.data-' in p.name for p in files) and any(p.suffix=='.index' for p in files)
        files.extend(p for p in folder.glob('*') if p.is_file())
        groups['6dof_'+model]={str(p.relative_to(root)):sha(p) for p in files}
    contact=root/'baseline_contact_graspnet/models/scene_test_2048_bs3_hor_sigma_001'
    groups['contact_graspnet']={str(p.relative_to(root)):sha(p) for p in
                               [*contact.glob('model.ckpt-54054.*'),contact/'config.yaml']}
    gg=root/'baseline_graspgen/models/checkpoints'
    config=gg/'graspgen_franka_panda.yml'
    # These filenames are explicit in the checked YAML and load_grasp_cfg;
    # the runtime hashes the resolved cfg.eval/discriminator checkpoint paths.
    groups['graspgen']={str(p.relative_to(root)):sha(p) for p in
                       [config,gg/'graspgen_franka_panda_gen.pth',gg/'graspgen_franka_panda_dis.pth']}
    record=dict(measured_at=datetime.now(timezone.utc).isoformat(),passed=True,groups=groups,
                model_inference_run=False,historical_weight_identity_proven=False,
                note='Current byte inventory; initial GPU log gaps are preserved, not retroactively overwritten.')
    (root/'validation/current_checkpoint_inventory.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(passed=True,groups={k:len(v) for k,v in groups.items()})))


if __name__=='__main__':main()
