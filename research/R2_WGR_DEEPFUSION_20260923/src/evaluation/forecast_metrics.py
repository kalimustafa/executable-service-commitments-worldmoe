import numpy as np
from vendor.r2_market import QUANTILES

def metrics_for(frame,model,seed,split,calibration,publication='pooled'):
    losses={};crossings=0
    for key,target in [('da','da_lbmp'),('dart','spread'),('rt','rt_integrated')]:
        q=frame[[f'{key}_q{t:.2f}' for t in QUANTILES]].to_numpy();y=frame[target].to_numpy()
        error=y[:,None]-q;tau=np.array(QUANTILES)
        losses[key]=np.maximum(tau*error,(tau-1)*error).mean(-1)
        crossings+=int((np.diff(q,axis=-1)<-1e-6).sum())
    q=frame[[f'rt_q{t:.2f}' for t in QUANTILES]].to_numpy();y=frame.rt_integrated.to_numpy()
    tail=frame.tail_event.to_numpy(bool)
    inside50=(y>=q[:,2])&(y<=q[:,4]);inside90=(y>=q[:,0])&(y<=q[:,6])
    score90=q[:,6]-q[:,0]+20*np.maximum(q[:,0]-y,0)+20*np.maximum(y-q[:,6],0)
    probs=frame[['regime_p0','regime_p1','regime_p2']].to_numpy();regime=frame.market_regime.to_numpy(int)
    return dict(model=model,seed=seed,split=split,calibration=calibration,publication=publication,rows=len(frame),tail_rows=int(tail.sum()),
        pinball=float(losses['rt'].mean()),rt_pinball=float(losses['rt'].mean()),dart_pinball=float(losses['dart'].mean()),da_pinball=float(losses['da'].mean()),
        overall_three_head_pinball=float(np.mean([x.mean() for x in losses.values()])),
        coverage50=float(inside50.mean()),coverage90=float(inside90.mean()),tail_coverage=float(inside90[tail].mean()) if tail.any() else np.nan,
        tail_pinball=float(losses['rt'][tail].mean()) if tail.any() else np.nan,tail_interval_score90=float(score90[tail].mean()) if tail.any() else np.nan,
        interval_width=float((q[:,6]-q[:,0]).mean()),width50=float((q[:,4]-q[:,2]).mean()),
        regime_accuracy=float((probs.argmax(-1)==regime).mean()),regime_log_loss=float(-np.log(probs[np.arange(len(frame)),regime].clip(1e-12)).mean()),
        quantile_crossing_count=crossings,mean_gate=float(frame.mean_gate.mean()))

def evaluate_frame(frame,model,seed,calibration):
    rows=[]
    for split,f in frame.groupby('split'):
        label='development_h2' if split=='exploratory_test' else split
        rows.append(metrics_for(f,model,seed,label,calibration))
        for pub,p in f.groupby('gate'):rows.append(metrics_for(p,model,seed,label,calibration,pub))
    economic=frame[(frame.timestamp_utc>='2024-08-21T00:00:00Z')&(frame.timestamp_utc<'2025-01-01T00:00:00Z')]
    rows.append(metrics_for(economic,model,seed,'economic_selection',calibration))
    return rows

