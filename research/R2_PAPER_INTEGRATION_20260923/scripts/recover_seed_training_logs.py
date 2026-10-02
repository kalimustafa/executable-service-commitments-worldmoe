"""Restore per-seed CSV logs from the complete original console JSON lines."""
from src.runtime import ROOT
import json
import csv
rows=[]
for seed in [20260921,20260922]:
    records=[]
    for line in (ROOT/'logs'/f'train_seed{seed}.log').read_text(encoding='utf-8-sig').splitlines():
        if not line.startswith('{'):continue
        record=json.loads(line)
        if record.get('seed')==seed and 'train_loss' in record:records.append(record)
    assert [r['epoch'] for r in records]==list(range(1,23))
    assert all(r['nonzero_residual_gradient'] for r in records)
    output=ROOT/'logs'/f'C1_ProfileResidual_seed{seed}_training.csv'
    with output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    complete=json.loads((ROOT/'logs'/f'C1_ProfileResidual_seed{seed}_completed.json').read_text())
    best=min([(complete['initial_price_score'],0)]+[(r['price_score'],r['epoch']) for r in records])
    assert best==(complete['best_price_score'],complete['best_epoch'])
    rows.append(dict(seed=seed,epochs=len(records),best=best,all_residual_gradients_nonzero=True,
        first_train_loss=records[0]['train_loss'],last_train_loss=records[-1]['train_loss'],
        best_trained_score=min(r['price_score'] for r in records),recovered_csv=str(output)))
(ROOT/'results/training_log_audit.json').write_text(json.dumps(dict(status='passed',
    log_naming_issue='The CSV basename omitted the seed; full per-seed console logs preserved all epochs. Reconstructed CSVs and corrected future basename.',
    model_weights_or_calibration_overwritten=False,rows=rows),indent=2))
print(json.dumps(rows,indent=2))
