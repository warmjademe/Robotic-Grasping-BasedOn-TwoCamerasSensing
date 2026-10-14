#!/usr/bin/env python3
"""Download and safely restore the immutable experiment archives."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def restore(archive, expected):
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert {m.name for m in members} == set(expected), 'Archive member list differs'
        assert len(members) == len(expected), 'Duplicate archive members'
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive member: '+member.name)
            dest = ROOT.joinpath(*path.parts)
            if not dest.resolve().is_relative_to(ROOT):
                raise ValueError('Destination leaves repository: '+member.name)
            record = expected[member.name]
            assert member.size == record['size']
            if dest.exists():
                if sha(dest) != record['sha256']:
                    raise FileExistsError('Existing data differs: '+str(dest))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name+'.download-part')
            with tar.extractfile(member) as source, temp.open('wb') as target:
                shutil.copyfileobj(source, target)
            assert sha(temp) == record['sha256'], member.name
            temp.replace(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', action='append', help='Archive name; repeat to select several')
    parser.add_argument('--cache', type=Path, default=ROOT/'.data-cache')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'tools/artifacts.json').read_text())
    names = {a['name'] for a in manifest['assets']}
    if args.asset and not set(args.asset) <= names:
        parser.error('Unknown archive. Available: '+', '.join(sorted(names)))
    args.cache.mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        if args.asset and asset['name'] not in args.asset:
            continue
        archive = args.cache/asset['name']
        if not archive.exists() or sha(archive) != asset['sha256']:
            part = archive.with_name(archive.name+'.part')
            print('Downloading', asset['name'], flush=True)
            with urllib.request.urlopen(asset['url'], timeout=120) as source, part.open('wb') as target:
                shutil.copyfileobj(source, target, 1024*1024)
            assert part.stat().st_size == asset['size'] and sha(part) == asset['sha256']
            part.replace(archive)
        group = asset['name'].removesuffix('.tar.gz')
        expected = {r['path']:r for r in manifest['files'] if r['storage'] == group}
        restore(archive, expected)
        print('Restored', asset['name'], len(expected), 'files', flush=True)


if __name__ == '__main__':
    main()
