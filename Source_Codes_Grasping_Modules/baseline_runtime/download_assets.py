#!/usr/bin/env python3
"""Download authors' public artifacts, retaining URLs, hashes, and failures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tarfile
import zipfile
import requests
import gdown

ROOT = Path(__file__).resolve().parent.parent
ASSETS = {
    'psmnet': ('baseline_psmnet_raft_stereo', 'file', '1xoqkQ2NXik1TML_FMUTNZJFAHrhLdKZG', 'psmnet_sceneflow.tar'),
    'raft': ('baseline_psmnet_raft_stereo', 'file', '1m3KoukUmKDoMv-ySOO6vBzYfWLyj9yqd', 'raftstereo-middlebury.pth'),
    'contact': ('baseline_contact_graspnet', 'folder', '1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl', ''),
    '6dof': ('baseline_6dof_graspnet', 'folder', '1DMYCz_-RoSZxGtydblpkSMHB9DTCIrkt', ''),
    'asgrasp': ('baseline_asgrasp', 'folder', '1omayRF-kl_HzkHRs9Ln7Dfn8L7dRUT9S', ''),
    'dexnet': ('baseline_dexnet2', 'https', 'https://berkeley.box.com/shared/static/j4k4z6077ytucxpo6wk1c5hwj47mmpux.zip', 'GQCNN-2.0.zip'),
}


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8388608), b''): h.update(chunk)
    return h.hexdigest()


def download_https(url, target):
    if target.exists() and target.stat().st_size: return
    part = target.with_suffix(target.suffix + '.partial')
    with requests.get(url, stream=True, timeout=(20, 90)) as r:
        r.raise_for_status()
        with part.open('wb') as f:
            for chunk in r.iter_content(1048576): f.write(chunk)
    part.rename(target)


def download(name):
    base, kind, identifier, filename = ASSETS[name]
    out = ROOT / base / 'models'
    out.mkdir(exist_ok=True)
    record = {'name': name, 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        if kind == 'folder':
            url = 'https://drive.google.com/drive/folders/' + identifier
            paths = gdown.download_folder(url, output=str(out), quiet=True, remaining_ok=False, skip_download=True)
            record['folder_listing'] = [str(p) for p in paths]
            paths = gdown.download_folder(url, output=str(out), quiet=True, remaining_ok=False)
            if not paths: raise RuntimeError('No files downloaded from author folder')
        elif kind == 'file':
            url = 'https://drive.google.com/uc?id=' + identifier
            target = out / filename
            if not target.exists():
                if not gdown.download(id=identifier, output=str(target), quiet=True):
                    raise RuntimeError('Author checkpoint download failed')
        else:
            url = identifier
            download_https(url, out / filename)
            if filename.endswith('.zip'):
                with zipfile.ZipFile(out / filename) as z:
                    for item in z.infolist():
                        assert (out / item.filename).resolve().is_relative_to(out.resolve())
                    z.extractall(out)
        record.update(status='downloaded', source_url=url,
                      files=[{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size,
                              'sha256': digest(p)} for p in sorted(out.rglob('*'))
                             if p.is_file() and p.name != name + '_download.json'])
    except Exception as exc:
        record.update(status='failed', error=str(exc))
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out / (name + '_download.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(name, record['status'], record.get('error', ''), flush=True)
    return record


def saxena():
    out = ROOT / 'baseline_saxena_2008' / 'upstream'
    out.mkdir(exist_ok=True)
    url = 'https://ai.stanford.edu/~asaxena/learninggrasp/code/'
    r = requests.get(url, timeout=30); r.raise_for_status()
    for filename in re.findall(r'href="([^"/]+\.(?:m|txt))"', r.text):
        download_https(url + filename, out / filename)
    download_https('https://ai.stanford.edu/~asaxena/learninggrasp/data/readme.txt', out / 'dataset_readme.txt')
    print('saxena', len(list(out.glob('*.m'))), 'original Matlab files downloaded', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--only', nargs='+', choices=list(ASSETS) + ['saxena'])
    a = p.parse_args(); names = a.only or list(ASSETS) + ['saxena']
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda n: saxena() if n == 'saxena' else download(n), names))
