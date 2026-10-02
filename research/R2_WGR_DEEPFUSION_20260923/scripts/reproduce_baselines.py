from src.runtime import ROOT, LEGACY
from src.data.access_guard import install_guard, read_development_parquet
install_guard()
import json
import time
import numpy as np
import torch
import pandas as pd
from src.data.r2_data import R2Data
from vendor.r2_market import R2MarketHead, R2Config, pinball, QUANTILES, identity_errors

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    data=R2Data();rows=[]
    for variant in ['no_world','full']:
        cp=torch.load(LEGACY/'checkpoints'/f'r2_{variant}_best.pt',weights_only=False,map_location='cpu')
        model=R2MarketHead(R2Config(**cp['model_config']),da_center=data.target_center[:,0],da_scale=data.target_scale[:,0],
                           dart_center=data.target_center[:,1],dart_scale=data.target_scale[:,1]).to(data.device)
        model.load_state_dict(cp['state_dict'],strict=True);model.eval()
        sums=torch.zeros(4,device=data.device);count=0
        with torch.inference_mode():
            for g in [0.,1.]:
                for ix in data.indices['selection'].split(4096):
                    labels,_=data.row_targets(ix);zone=data.zone[ix];pub=torch.full((len(ix),),g,device=data.device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),pub,labels[:,0])
                    losses=[pinball(out[key],labels[:,j])/data.target_scale[zone,j] for j,key in enumerate(['da','dart','rt'])]
                    sums+=torch.stack([x.sum() for x in losses]+[(.5*losses[0]+losses[1]+.5*losses[2]).sum()]);count+=len(ix)
            reference=read_development_parquet(LEGACY/'forecast_development'/f'R2_{variant}_raw.parquet')
            indexes=data.frame.reset_index().set_index(['timestamp_utc','zone'])['index']
            differences=[];ours=[];original=[];identity=[]
            for g,label in [(0.,'pre_DA'),(1.,'post_DA')]:
                ref=reference[reference.gate.eq(label)]
                ids=indexes.reindex(pd.MultiIndex.from_frame(ref[['timestamp_utc','zone']])).to_numpy()
                for start in range(0,len(ids),4096):
                    ix=torch.tensor(ids[start:start+4096],device=data.device);y=data.labels[ix];pub=torch.full((len(ix),),g,device=data.device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ix),pub,y[:,0])
                    identity.append(identity_errors(out,pub,y[:,0]))
                    f=ref.iloc[start:start+len(ix)]
                    for j,key in enumerate(['da','dart','rt']):
                        q=f[[f'{key}_q{t:.2f}' for t in QUANTILES]].to_numpy()
                        pred=out[key].cpu().numpy();differences.append(float(np.max(np.abs(pred-q))))
                        residual=y[:,j].cpu().numpy()[:,None]-q
                        original.extend(np.maximum(np.array(QUANTILES)*residual,(np.array(QUANTILES)-1)*residual).mean(-1).tolist())
                        ours.extend(pinball(out[key],y[:,j]).cpu().tolist())
        score=float((sums/count)[3]);saved=float(cp['validation']['val_price_score'])
        row=dict(model=variant,epoch=cp['epoch'],saved_selection_score=saved,reproduced_selection_score=score,
                 selection_absolute_error=abs(score-saved),saved_development_pinball=float(np.mean(original)),
                 reproduced_development_pinball=float(np.mean(ours)),max_quantile_absolute_difference=max(differences),
                 max_identity_error=max(max(r.values()) for r in identity),runtime=torch.__version__)
        # BF16 + runtime changes can change individual quantiles; aggregate
        # reproduction must remain within 0.2%, with no identity violation.
        row['passed']=abs(score-saved)<=max(1e-5,.002*abs(saved)) and abs(np.mean(ours)-np.mean(original))<=.002*abs(np.mean(original)) and row['max_identity_error']<1e-3
        rows.append(row);print(json.dumps(row),flush=True)
        del model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(ROOT/'results/baseline_reproduction_2024.csv',index=False)
    assert all(row['passed'] for row in rows), 'Baseline reproduction failed; fix before deep fusion.'
    (ROOT/'results/baseline_reproduction_passed.json').write_text(json.dumps(dict(status='passed',details=rows,locked_targets_accessed=False),indent=2))

if __name__=='__main__':main()
