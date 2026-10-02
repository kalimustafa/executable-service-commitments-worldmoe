"""Freeze the selected development-only R2 configuration before 2025 scoring."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    selected=json.loads((ROOT/'validation_economics/selected_calibration.json').read_text())
    ood=json.loads((ROOT/'results/w1_ood_completed.json').read_text())
    assert selected['evaluation_2025_opened'] is False and ood['status']=='complete'
    paths=[ROOT/'checkpoints'/name for name in ['r2_full_best.pt','r2_no_world_best.pt','r2_full_native_calibration.pt','r2_no_world_native_calibration.pt',
        'r2_full_external_calibration.json','r2_no_world_external_calibration.json']]
    paths += [ROOT/'cache/r2_normalization.json']
    inventory={p.name:dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths}
    design=dict(schema='r2.ef.frozen_retrospective_evaluation.v1',status='frozen_before_R2_2025_evaluation',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),selected_world_seed=20240829,
        primary_model=selected['primary_model'],primary_calibration=selected['primary_calibration'],primary_interval=50,
        windows_h=[7,12,24],mixed_deadlines=[dict(fraction=.25,hours=4),dict(fraction=.5,hours=12),dict(fraction=.25,hours=24)],
        slot_head_enabled=False,ranking='forecast RT median / central-50% interval',
        action_adapter=False,world_input_contract='frozen issued reference context, not live EF state',
        W1_live_state_ood_grade=ood['grade'],trainable_W0=0,trainable_W1=0,
        evaluate_variants=['full_raw','full_native','full_external','no_world_native'],
        matching_policies=['published_DA','old_RASC_point','old_RASC_interval','R2_point','R2_interval','R2_no_world_point','R2_no_world_interval'],
        neural_baseline='associated old R on selected seed20240829; legacy seed20240813 US$395 retained separately as historical anchor',
        historical_2025_previously_used=True,R2_training_used_2025=False,R2_calibration_used_2025=False,
        post_evaluation_parameter_changes_allowed=False,files=inventory)
    file=ROOT/'EVALUATION_FREEZE.json'
    if file.exists():
        prior=json.loads(file.read_text());assert prior['files']==inventory;print('Existing freeze retained');return
    file.write_text(json.dumps(design,ensure_ascii=False,indent=2),encoding='utf-8')
    marker=dict(schema='dcvpp.r_ecmoe.frozen_evaluation_marker.v1',status='FROZEN_R2_RETROSPECTIVE_EVALUATION',final_year=2025,
        model_selection_frozen=True,thresholds_frozen=True,data_statistics_frozen=True,locked_final_opened=False,
        expected_evaluable_days=362,one_shot_id='r2-ef-seed20240829',prior_2025_already_evaluated=True,
        interpretation='New R2 evaluation after development freeze; not a pristine project-level holdout')
    (ROOT/'cache/frozen_evaluation_marker.json').write_text(json.dumps(marker,indent=2))
    print('Frozen',design['primary_model'],'before new 2025 evaluation')
if __name__=='__main__':main()
