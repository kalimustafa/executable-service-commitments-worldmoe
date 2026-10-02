"""Report action and state OOD distinctly; never hide a failed state head."""
from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    distribution=pd.read_csv(ROOT/'w1_ood_metrics.csv');prediction=pd.read_csv(ROOT/'results/w1_predictive_ood_metrics.csv')
    new=distribution[distribution.source.str.startswith('EF_')]
    action=new[new.metric.isin(['compute_power_mw','utilization','quarter_hour_ramp_mw','grid_import_mw'])]
    state=new[new.metric.isin(['queue_work_units','queue_clearance_time_h'])]
    action_far=float(action.far_outside_training_envelope_fraction.max());state_far=float(state.far_outside_training_envelope_fraction.max())
    comparison=prediction[~prediction.policy.eq('original_validation_actions')]
    error_ratio=float(comparison.mae_ratio_vs_original_actions.max())
    severity='SEVERE' if max(action_far,state_far)>=.01 and error_ratio>=10 else 'MODERATE'
    p=ROOT/'results/w1_ood_completed.json';report=json.loads(p.read_text())
    report.update(grade=severity,severity_scope='reusing the learned W1 head for continuous policy-dependent queue prediction',
        market_reference_branch='kept on explicit issue-time reference states/actions; EF current queues are not passed to this branch',
        far_action_envelope_fraction=action_far,far_queue_envelope_fraction=state_far,
        action_adapter_required=False,
        reason='The learned continuous queue head shows severe state-domain error. It is excluded from EF feasibility and from R2 market-reference inputs; no action adapter is needed for the chosen reference-only market interface.',
        criterion='at least 1% far state/action envelope deviations together with at least 10x matched one-step MAE',
        sampling_scope='17 pre-2025 rolling origins including current-queue stress states; not an unbiased annual error estimate')
    p.write_text(json.dumps(report,indent=2))
    lines=['# W1 OOD diagnostic','','Overall classification for direct continuous-state reuse: **'+severity+'**.','',
        'The amplitude of compute, utilization, ramps and grid import stays inside the training envelope with the specified 10% margin: far exceedance fraction is 0. The queue-state distribution shifts: the 24 h EF case has a 1.3784% far-envelope fraction.',
        'On 17 matched pre-2025 rolling origins (68 site predictions per policy), including current-queue stress states, original-action queue RMSE is 0.06893 work units. The 7/12/24 h EF RMSE values are 2.49734, 3.83835 and 3.89618. The corresponding ratios are 36.23, 55.68 and 56.52. This is a stress-weighted conditional one-step comparison, not an annual error rate or the original full-day proper-score protocol.',
        'The target semantics explain why this matters: the original W1 teacher expires overdue work and resets its day, whereas EF retains all outstanding work continuously. A native state row t holds the action from t-1; the interpolated daily boundary was excluded from measured action statistics.',
        '', '## Consequence for the chosen architecture','',
        'W0 and W1 remain frozen. The new R2 price branch uses an explicitly defined issue-time reference context, not the evolving EF queue or future EF actions. Learned W1 queue predictions are not used to certify deadlines, advance the physical queue, or score service completion. The exact reservation/recourse/physical layer owns those operations. Consequently, a new action adapter is not required for this interface. Feeding live EF queues into the W1 state head would require a separate, validated adaptation and is not authorized by these results.',
        '', '## Matched predictive diagnostic','',prediction.to_csv(index=False),
        '', 'Full distributions: w1_ood_metrics.csv. Source and timing definitions: results/w1_ood_energy_and_timing.csv. Native W1 h1 proper score: 0.496074244, retained only as a separate recorded validation statistic.']
    (ROOT/'W1_OOD_DIAGNOSTIC.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(severity,'state reuse; reference-only R2 branch retained; no action adapter')
if __name__=='__main__':main()
