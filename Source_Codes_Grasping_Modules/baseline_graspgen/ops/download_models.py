#!/usr/bin/env python3
"""Download pinned public GraspGen assets; no credentials required."""
import concurrent.futures
import hashlib
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
REVISION = "ec1ccbb5eec0680db669246ac312a3636f16ee43"
REPO = "adithyamurali/GraspGenModels"

def download(item):
    name = item['path']
    target = ROOT / 'models' / name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
    expected = item.get('lfs', {}).get('oid')
    for attempt in range(4):
        try:
            if not target.exists() or target.stat().st_size != item['size']:
                temp = target.with_name(target.name + '.part')
                with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
                assert temp.stat().st_size == item['size'], (name, temp.stat().st_size, item['size'])
                temp.replace(target)
            hasher = hashlib.sha256()
            with target.open('rb') as src:
                while block := src.read(1024 * 1024):
                    hasher.update(block)
            digest = hasher.hexdigest()
            assert not expected or expected == digest, (name, 'hash mismatch')
            print(f'OK {name} {target.stat().st_size}', flush=True)
            return dict(path=name, bytes=target.stat().st_size, sha256=digest, url=url)
        except Exception as exc:
            print(f'Retry {attempt + 1}: {name}: {type(exc).__name__}', flush=True)
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))

if __name__ == '__main__':
    listing_url = f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true'
    listing = json.load(urllib.request.urlopen(listing_url, timeout=60))
    selected = [x for x in listing if x['type'] == 'file' and (
        x['path'].startswith(('sample_data/real_object_pc/', 'sample_data/meshes/')) or
        x['path'].startswith(('checkpoints/graspgen_robotiq_2f_140', 'checkpoints/graspgen_franka_panda')) or
        x['path'] in ('README.md', 'LICENSE'))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, selected))
    (ROOT / 'models' / 'manifest.json').write_text(json.dumps(
        dict(repository=REPO, revision=REVISION, files=files), indent=2) + '\n')
