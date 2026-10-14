"""Keep RQ exports current while the two physical backends finish."""
import fcntl
import json
import time
from common import RUN, STUDY, sha, write
from summarize import summarize

if __name__=='__main__':
    lock=(RUN/'ops/summary.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        status=summarize()
        if status['complete']:
            manifest=json.loads((RUN/'inputs/manifest.json').read_text())
            mismatches=[rel for rel,expected in manifest['source_sha256'].items() if sha(STUDY/rel)!=expected]
            write(RUN/'analysis/input_verification.json',dict(checked=len(manifest['source_sha256']),mismatches=mismatches,
                completed_unix=time.time(),valid=not mismatches))
            if mismatches:raise RuntimeError('Frozen input hash mismatch')
            break
        time.sleep(30)
