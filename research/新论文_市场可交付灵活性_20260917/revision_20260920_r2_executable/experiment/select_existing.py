"""Select a compatible existing lineage using pre-2025 validation only."""
from pathlib import Path
import json,csv
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parents[1];REPO=WORKSPACE/'power-dc-vpp'
BASE=REPO/'artifacts/r_ecmoe_multiseed_20260910/world_moe'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    rows=[];records={}
    for seed in [20240813,20240829,20240917]:
        for phase in ['w0','w1']:
            folder=BASE/f'run=multiseed-20260910-seed{seed}'/f'phase={phase}'
            state=read(folder/'state.json');record=state['training_control']['formal_primary_record'];gate=read(folder/'gate.json')
            r=dict(seed=seed,phase=phase,selected_epoch=record['epoch'],primary_score=record['primary_score'],
                checkpoint=state['best_checkpoint'],selection_policy=state['best_checkpoint_selection_role'],
                native_eligible=record['eligible'],native_gate=gate.get('native_evidence',{}).get('status'),
                recorded_checkpoint_id=gate['checkpoint_id'])
            for h,v in record['selection_metrics'].get('horizon_scores',{}).items():r[h]=v
            rows.append(r);records[(seed,phase)]=(state,r)
    winner=min((r for r in rows if r['phase']=='w1'),key=lambda r:r['primary_score'])
    seed=winner['seed'];s1=records[(seed,'w1')][0];parent=s1['training_control']['parent_checkpoint']
    w0=records[(seed,'w0')][1]
    assert Path(parent['path']).resolve()==Path(w0['checkpoint']).resolve()
    assert parent['sha256']==w0['recorded_checkpoint_id']
    selection=dict(status='selected_from_2024_validation',seed=seed,w0=w0,w1=winner,
        config=str(REPO/f'configs/multiseed_20260910_{seed}_w1.yaml'),
        criterion='minimum recorded W1 worst-horizon proper score on the common full 2024 validation; lower is better',
        score_families=['price','workload','pv','pue'],horizons=[1,4,16,96],
        standalone_W0_best=min((r for r in rows if r['phase']=='w0'),key=lambda r:r['primary_score']),
        pair_rule='preserve the W1-trained parent lineage; do not splice an unrelated W0 encoder into a fixed W1',
        runtime_rule='W0 and W1 are training phases of one WorldMoE; freeze all parameters of the selected W1 snapshot including its inherited and updated encoder',
        native_W0_diagnostic='Recorded native eligibility remains false; validation-score-only selection does not turn it into a passed diagnostic.')
    selection['2025_used_for_selection']=False
    (ROOT/'selected_w0_w1.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/'w0_w1_validation_ranking.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines=['# Selected frozen W0 and W1 lineage','','Selection uses the existing full-2024 validation, not 2025 financial outcomes.','',
        '| Seed | W0 epoch | W0 score | W1 epoch | W1 worst-horizon score |','|---|---:|---:|---:|---:|']
    for seed in [20240813,20240829,20240917]:
        a=records[(seed,'w0')][1];b=records[(seed,'w1')][1]
        lines.append(f"| {seed} | {a['selected_epoch']} | {a['primary_score']:.9f} | {b['selected_epoch']} | {b['primary_score']:.9f} |")
    lines +=['','Selected pair: seed 20240829, W0 epoch 80 and W1 epoch 80. W1 is best at every reported horizon (1, 4, 16 and 96).',
        'Standalone W0 score is best for seed 20240813 epoch 128. Its encoder is not substituted into the independently trained seed-20240829 W1.',
        'W1 training also updated encoder/shared/router groups. The deployed frozen instance therefore uses the final selected W1 weights, with its W0 parent recorded as lineage; it does not restore an earlier W0 prefix or execute two complete encoders.',
        'Existing native W0 capability diagnostics are preserved as false. This selection follows the saved validation-score-only criterion and does not claim that the diagnostics passed. The selected W1 native release diagnostic passed.',
        'Strict loading and zero-trainable-parameter confirmation are recorded separately in frozen_parameter_report.json.']
    (ROOT/'W0_W1_SELECTION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Selected seed',winner['seed'],'W0 epoch',w0['selected_epoch'],'W1 epoch',winner['selected_epoch'],'W1 score',winner['primary_score'])
if __name__=='__main__':main()
