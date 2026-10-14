"""One-shot host commissioning; wait for unrelated GPU workloads to finish."""
import fcntl
import json
import os
import subprocess
import time
from common import RUN, PROTOCOL, write
from host_probe import probe
from physics_batch import calibrated, BASE

if __name__=='__main__':
    lock=(RUN/'ops/commissioning.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LD_LIBRARY_PATH=str(BASE/'.sim-libs/lib'))
    for engine in ['mujoco','isaac']:
        for profile in PROTOCOL['physics']['profiles']:
            if calibrated(engine,profile):continue
            while not probe()['hardware_gpu_ready']:
                write(RUN/'ops/host_commissioning.json',dict(status='waiting_for_idle_host',engine=engine,profile=profile,resources=probe()))
                time.sleep(15)
            dest=RUN/'calibration'/f'{engine}_{profile}_v2'
            if engine=='mujoco' and any(dest.glob('*/result.json')):raise RuntimeError('Partial calibration needs diagnosis')
            if engine=='isaac' and (dest/'result.json').exists():raise RuntimeError('Failed calibration needs diagnosis')
            exe=BASE/('.venv-sim' if engine=='isaac' else '.venv-mujoco')/'bin/python'
            with (RUN/'logs'/f'host_commissioning_{engine}_{profile}.log').open('wb') as log:
                r=subprocess.run([str(exe),str(RUN/'code'/f'run_{engine}.py'),'--tasks','calibration','--profile',profile,'--output',str(dest)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            if r.returncode or not calibrated(engine,profile):
                write(RUN/'ops/host_commissioning.json',dict(status='calibration_failed',engine=engine,profile=profile,returncode=r.returncode))
                raise RuntimeError('Calibration failed; preserve evidence')
    write(RUN/'ops/host_commissioning.json',dict(status='validated',resources=probe()))
