"""Read-only resource check; shared GPU lock protects other robot experiments."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
def probe():
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    gpu=[]
    cmd=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    for line in cmd.stdout.strip().splitlines():
        pid,used=line.split(',');pid=int(pid)
        try:name=Path(f'/proc/{pid}/comm').read_text().strip()
        except FileNotFoundError:continue
        try:args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError,PermissionError):args=[]
        project_job=any(str(ROOT/'platform_comparison_20261014/code/run_isaac.py')==a for a in args)
        gpu.append(dict(pid=pid,memory_mib=int(used),name=name,display_process=name in ['gnome-remote-de','gnome-shell','Xorg'],project_job=project_job))
    held=False
    path=ROOT/'physics_validation_20261013/gpu.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:held=True
    cpus=os.cpu_count();load=os.getloadavg()[0]
    free=shutil.disk_usage(ROOT).free
    cpu_ready=load<cpus*.8 and memory['MemAvailable']>12*1024**3 and free>30*1024**3
    calibration={}
    for engine in ['isaac','mujoco']:
        valid=True
        for profile in ['nominal','friction_05']:
            folder=ROOT/'platform_comparison_20261014/calibration'/f'{engine}_{profile}_v2'
            files=[folder/'result.json'] if engine=='isaac' else list(folder.glob('*/result.json'))
            rows=[]
            for f in files:
                if f.exists(): rows+=json.loads(f.read_text())['rows']
            valid=valid and len(rows)==4 and all(r['execution_success']==r['expected'] for r in rows)
        calibration[engine]=valid
    hardware_gpu_ready=cpu_ready and cmd.returncode==0 and not held and not any(not p['display_process'] for p in gpu)
    return dict(host=socket.gethostname(),unix=time.time(),cpu_count=cpus,load1=load,
        memory_available_gib=memory['MemAvailable']/1024**3,disk_free_gib=free/1024**3,
        gpu_processes=gpu,gpu_lock_held=held,cpu_ready=cpu_ready,
        calibration_ready=calibration,hardware_gpu_ready=hardware_gpu_ready,
        cpu_experiment_ready=cpu_ready and not any(not(p['display_process'] or p['project_job']) for p in gpu),
        gpu_ready=hardware_gpu_ready and all(calibration.values()))

if __name__=='__main__':print(json.dumps(probe()))
