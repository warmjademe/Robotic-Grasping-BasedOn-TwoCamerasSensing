#!/usr/bin/env python3
"""Verify bytes and symlinks against the release manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'tools/artifacts.json').read_text())
    errors = []
    count = 0
    for row in manifest['files']:
        if args.git_only and row['storage'] != 'git':continue
        path = ROOT/row['path']
        if not path.is_file():
            errors.append('missing: '+row['path']);continue
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != row['sha256'] or path.stat().st_size != row['size']:
            errors.append('changed: '+row['path'])
        count += 1
    for rel, target in manifest['symlinks'].items():
        path = ROOT/rel
        if not path.is_symlink() or os.readlink(path) != target:
            errors.append('symlink differs: '+rel)
        elif not args.git_only and not path.exists():errors.append('broken symlink: '+rel)
    print(json.dumps(dict(checked_files=count, errors=errors), indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':main()
