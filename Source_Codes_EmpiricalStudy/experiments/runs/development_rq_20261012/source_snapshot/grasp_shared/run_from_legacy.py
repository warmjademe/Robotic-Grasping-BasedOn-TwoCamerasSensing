#!/usr/bin/env python3
"""One-call bridge from a saved reconstruction folder to the common backends."""
import argparse
import subprocess
import sys
from pathlib import Path
from export_legacy_cloud import export
from infer import METHOD_IDS

p=argparse.ArgumentParser()
p.add_argument('--method',choices=METHOD_IDS,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--cameras',type=Path)
p.add_argument('--reference')
a=p.parse_args()
if a.out.exists():raise FileExistsError('Use a new bridge output directory')
a.out.mkdir(parents=True)
observation=a.out/'observation.npz'
export(a.source,observation,a.cameras,a.reference)
subprocess.run([sys.executable,str(Path(__file__).with_name('infer.py')),
                '--method',a.method,'--input',str(observation),'--out',str(a.out/'prediction')],check=True)
