"""Hash the retained new verification materials without changing results."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
files=[]
for p in sorted(root.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='new_verification_manifest.json':continue
    files.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,
                  'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
terminal=[]
for name in ['terminal_rate_audit.py','terminal_rate_audit.json','terminal_rate_audit.txt']:
    p=root.parent/name
    if p.exists():terminal.append({'path':'../'+name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
out={'title':'New conservative annual envelope certificates, 31 October 2026',
     'original_source_commit':'323c2084e7ee668ba512fb1e33dd56b8394b32d4',
     'annual_input_sha256':'5be123c40f01b111cf6503b07882e703a4c2cc29a03dde3cfb7aa4321ce2cf72',
     'scope':{'new_original_cost_annual_certificates':12,'new_epigraph_annual_certificates':12,
              'exact_robust_surrogate_schedules':4,'actual_source_admissions_recovered':False,
              'admission_specific_actual_witness_materialized':False,
              'uniform_actual_admission_repair_existence_proved':True,
              'original_29oct_certificates_reproduced':False,'old_117000_dollar_witness_recovered':False,
              'reference_account':'Fixed published reference expense and fee anchor',
              'terminal_account':'Signed tau*(R_N-X_N), with 21*E allowance per site',
              'first_release_causal_replay_performed':False},
     'terminal_sign_enclosure_records':terminal,'files':files}
(root/'new_verification_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'files':len(files),'total_bytes':sum(v['bytes'] for v in files)},indent=2))
