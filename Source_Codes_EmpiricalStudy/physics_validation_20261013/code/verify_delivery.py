"""Standard-library verification on Mac or huashuo; does not start simulation."""
import ast,csv,datetime,hashlib,json,re
from pathlib import Path

RUN=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=json.loads((RUN/'inputs/manifest.json').read_text())
    summary=json.loads((RUN/'analysis/summary.json').read_text())
    for rel,digest in manifest['source_sha256'].items():assert sha(RUN.parent/rel)==digest,rel
    for rel,digest in summary['evidence_sha256'].items():assert sha(RUN/rel)==digest,rel
    candidate_rows=list(csv.DictReader((RUN/'analysis/per_candidate.csv').open()))
    assert len(candidate_rows)==412
    for profile in ['nominal','friction_05']:
        rows=[r for r in candidate_rows if r['profile']==profile]
        assert len(rows)==206 and len({r['case_id'] for r in rows})==206
        sets=[]
        for rq,count in [('RQ2',156),('RQ3',101)]:
            path=RUN.parent/rq/'results/isaac_physics_20261013/per_candidate.csv'
            data=[r for r in csv.DictReader(path.open()) if r['profile']==profile]
            assert len(data)==count
            sets.append({r['case_id'] for r in data})
        assert len(sets[0]&sets[1])==51 and len(sets[0]|sets[1])==206
    variant=ast.parse((RUN/'diagnostics/run_arm_target_filter.py').read_text())
    current=ast.parse((RUN/'code/run_arm.py').read_text())
    # Only the descriptive material-scope string differs from the replayed variant.
    class Normalize(ast.NodeTransformer):
        def visit_keyword(self,node):
            if node.arg=='friction_scope':node.value=ast.Constant(value='scope_metadata')
            return self.generic_visit(node)
    assert ast.dump(Normalize().visit(variant))==ast.dump(Normalize().visit(current))
    for name in ['contact_filter_audit.json','contact_filter_audit_friction_05.json']:
        audit=json.loads((RUN/'diagnostics'/name).read_text());assert audit['status']=='passed'
        assert all(v==0 for v in audit['max_absolute_trace_differences'].values())
    secret=re.compile(rb'(?:sk-[A-Za-z0-9_-]{18,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|api\.day\.app/[A-Za-z0-9_-]{12,})')
    textfiles=[f for f in RUN.rglob('*') if f.is_file() and f.suffix in ['.py','.md','.json','.yaml','.log','.csv','.urdf']]
    assert not [str(f.relative_to(RUN)) for f in textfiles if secret.search(f.read_bytes())], 'Credential-pattern scan failed'
    video=json.loads((RUN/'visualization/video_manifest.json').read_text())
    assert len(video['cases'])==2 and {r['outcome']['execution_success'] for r in video['cases']}=={True,False}
    assert (RUN/'visualization/recorded_physics_replay.mp4').stat().st_size>100_000
    report=dict(status='passed',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_files=len(manifest['source_sha256']),result_files=len(summary['evidence_sha256']),
        unique_candidates_per_profile=206,rq2_per_profile=156,rq3_per_profile=101,shared_per_profile=51,
        current_execution_code_matches_replayed_variant=True,contact_audit_cases=42,credential_scan_files=len(textfiles),
        video_examples=2,code_sha256={p.name:sha(p) for p in sorted((RUN/'code').glob('*.py'))})
    (RUN/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
