"""Restore the highest measured eligible saving, independently of min-delta patience."""
import json,shutil,argparse,hashlib
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--prefix',default='DeepR2',choices=['DeepR2','QPR2','DistR2']);a=p.parse_args()
label={'DeepR2':'deep','QPR2':'qp','DistR2':'distribution'}[a.prefix];summary_file=root/'results'/f'{label}_primary_complete.json'
summary=json.loads(summary_file.read_text());reference=torch.load(root/'checkpoints/StructuredR2_seed20260920_epoch00.pt',map_location='cpu',weights_only=False)['metrics']['price_score']
eligible=[]
for path in (root/'results').glob(f'{a.prefix}_Eagle_2024_s20260920_ep*.json'):
    row=json.loads(path.read_text())
    if row.get('oracle_diagnostic') or row.get('eligible_for_80_gate') is False:continue
    epoch=row['model_epoch'];checkpoint=root/'checkpoints'/f'{a.prefix}_seed20260920_epoch{epoch:02d}.pt'
    metrics=torch.load(checkpoint,map_location='cpu',weights_only=False)['metrics']
    if row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8 and metrics['price_score']<=reference*1.02:eligible.append(row)
best=max(eligible,key=lambda r:r['saving_usd']);source=root/'checkpoints'/f'{a.prefix}_seed20260920_epoch{best["model_epoch"]:02d}.pt';destination=root/'checkpoints'/f'{a.prefix}_seed20260920_best.pt';shutil.copyfile(source,destination)
summary.update(best_epoch=best['model_epoch'],best_2024_savings_usd=best['saving_usd'],capture_fraction=best['capture_lower_fraction'],target_reached=best['capture_lower_fraction']>=.8,best_restored=True,selection='Maximum eligible actual annual saving; 50 USD min-delta controls patience only',checkpoint_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
summary_file.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
