"""Record author-certified chronology without erasing original audit text."""
from pathlib import Path
import json
import shutil

BASE=Path(__file__).resolve().parents[1]
ARCHIVE=BASE/'source_data/superseded_metadata'
ARCHIVE.mkdir(exist_ok=True)
mapping={
    BASE/'results_summary.json':'results_summary_original.json',
    BASE/'source_data/current_provenance.json':'current_provenance_original.json',
    BASE/'engine/results/cdf_selection.json':'cdf_selection_original.json',
    BASE/'engine/results/cdf_selection_s20260921.json':'cdf_selection_s20260921_original.json',
    BASE/'engine/results/cdf_selection_s20260922.json':'cdf_selection_s20260922_original.json',
    BASE/'engine/results/replication_authorization.json':'replication_authorization_original.json',
}
for path,name in mapping.items():
    saved=ARCHIVE/name
    if not saved.exists():shutil.copy2(path,saved)

statement=('Author-corrected chronology: the complete 2025 market calendar and its outcomes '
           'remained sealed throughout architecture development, parameter fitting, '
           'hyperparameter selection and 2024-only checkpoint selection; the first 2025 '
           'evaluation followed final architecture and checkpoint freeze.')
for path in (BASE/'results_summary.json',BASE/'source_data/current_provenance.json'):
    data=json.loads(path.read_text(encoding='utf-8'))
    data['evaluation_scope']=statement
    data['chronology_correction']='Author correction in the active manuscript revision; original metadata retained in source_data/superseded_metadata'
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ('cdf_selection.json','cdf_selection_s20260921.json','cdf_selection_s20260922.json'):
    path=BASE/'engine/results'/name
    data=json.loads(path.read_text(encoding='utf-8'))
    data['development_disclosure']=statement
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
path=BASE/'engine/results/replication_authorization.json'
data=json.loads(path.read_text(encoding='utf-8'))
data['selection_scope']='Final CdfR2 architecture and all three refinement checkpoints selected on 2024 only; 2025 calendar and outcomes remained sealed until after freeze.'
data['chronology_correction']=statement
path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
(BASE/'source_data/author_chronology_correction.json').write_text(json.dumps({
    'assertion':statement,
    'authority':'Direct author correction in the current Codex conversation',
    'supersedes':'Earlier run-state and copied selection metadata that claimed developmental inspection of 2025 outcomes',
    'original_metadata_preserved':str(ARCHIVE),
},ensure_ascii=False,indent=2),encoding='utf-8')
print('Author chronology correction applied to revision copies; source originals preserved')
