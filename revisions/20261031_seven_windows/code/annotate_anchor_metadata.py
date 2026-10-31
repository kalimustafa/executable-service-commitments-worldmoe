"""Clarify an initial metadata sentence without changing computations or logs."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
OLD='all admitted obligations retained with their actual deadline; complete queue enters deadline prefix guard and continuation'
NEW='all obligations retained; year-end early-completion target only; original exclusive deadline retained for independent audit; complete queue enters prefix guard and continuation'
for p in (ROOT/'results').glob('*_run_manifest.json'):
    r=json.loads(p.read_text(encoding='utf-8'))
    if r.get('status')=='complete' and r.get('queue_rule')==OLD:
        r['metadata_clarification']={'original_queue_rule_text':OLD,'reason':'The original year-end rule imposes an earlier policy target, not a changed contractual deadline. No calculation or stored trajectory is changed.','date':'2026-10-31','executed_source_copy':'code/executed_runner_initial_anchor.py'}
        r['queue_rule']=NEW;p.write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(p.name)
