#!/usr/bin/env python3
"""Fetch upstream source at recorded commits; refuse to overwrite existing trees."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',action='append',help='Upstream directory name; omit for all')
    args=parser.parse_args()
    records=json.loads((ROOT/'tools/upstreams.json').read_text())
    names={Path(r['path']).name for r in records}
    if args.name and not set(args.name)<=names:parser.error('Unknown upstream')
    for row in records:
        dest=ROOT/row['path']
        if args.name and dest.name not in args.name:continue
        if dest.exists():raise FileExistsError(str(dest))
        dest.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',row['url'],str(dest)],check=True)
        subprocess.run(['git','-C',str(dest),'checkout','--detach',row['commit']],check=True)
        subprocess.run(['git','-C',str(dest),'submodule','update','--init','--recursive'],check=True)
        if row.get('patch'):
            subprocess.run(['git','-C',str(dest),'apply',str(ROOT/row['patch'])],check=True)
        print(dest.name,row['commit'])


if __name__=='__main__':main()
