"""Export frozen, issue-specific forecasts and shared features for 2024 replay."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import *
import numpy as np,pandas as pd,torch
from src.data.r2_data import R2Data,ZONES
from src.models.factory import make_model
from src.models.joint_decision import JointDecision

p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=20260920);p.add_argument('--epochs',type=int,nargs='+');args=p.parse_args()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
data=R2Data(device='cuda',multiscale=True)
with np.load(WORKSPACE/'external_validation/data/forcing_2024.npz') as z:times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
hours=times.floor('h').unique()
index=pd.MultiIndex.from_product([hours,ZONES],names=['timestamp_utc','zone'])
frame=data.frame.copy();frame.timestamp_utc=pd.to_datetime(frame.timestamp_utc,utc=True)
lookup=pd.Series(np.arange(len(frame)),index=pd.MultiIndex.from_frame(frame[['timestamp_utc','zone']]))
indices=torch.tensor(lookup.reindex(index).to_numpy(np.int64),device=data.device)
assert set(frame.iloc[indices.cpu().numpy()].date.str[:4])=={'2024'}
if args.epochs is None:
    logs=pd.read_csv(ROOT/'logs'/f'joint_training_seed{args.seed}.csv')
    initial=torch.load(ROOT/'checkpoints'/f'JointR2_seed{args.seed}_epoch00.pt',map_location='cpu',weights_only=False)['metrics']
    eligible=logs[logs.price_score<=initial['price_score']*1.02]
    assert len(eligible),'No checkpoint meets the 2% forecast-score degradation screening guard'
    best=int(eligible.sort_values(['ranking_regret','price_score']).iloc[0].epoch)
    epochs=[0,best]
else:epochs=args.epochs
for epoch in epochs:
    checkpoint=ROOT/'checkpoints'/f'JointR2_seed{args.seed}_epoch{epoch:02d}.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    base=make_model('C1_ProfileResidual',args.seed,data)
    model=JointDecision(base,data.target_scale,data.target_center[:,2],torch.ones(4,device=data.device)).cuda()
    model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
    arrays={'hours':hours.asi8,'pue_reference':model.pue_reference.cpu().numpy(),'historical_rt':model.historical_rt.cpu().numpy()}
    identity=0.
    with torch.inference_mode():
        for gate in [0,1]:
            values=[];latents=[]
            for ids in indices.split(4096):
                y=data.labels[ids]
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ids),torch.full((len(ids),),float(gate),device=data.device),y[:,0])
                assert torch.diff(out['rt'],dim=-1).min()>=-1e-5
                if gate:identity=max(identity,float((out['rt']-(y[:,0,None]-out['dart'].flip(-1))).abs().max()))
                values.append(out['rt'].cpu().float().numpy());latents.append(out['decision_latent'].cpu().float().numpy())
            arrays[f'rt_{gate}']=np.concatenate(values).reshape(len(hours),4,7)
            arrays[f'latent_{gate}']=np.concatenate(latents).reshape(len(hours),4,128)
    output=ROOT/'cache'/f'market2024_s{args.seed}_ep{epoch:02d}.npz'
    np.savez_compressed(output,**arrays)
    (output.with_suffix('.json')).write_text(json.dumps(dict(seed=args.seed,epoch=epoch,checkpoint=str(checkpoint),forecast_rows=len(indices)*2,apic_max_error=identity,selection_rule='Minimum held-out 2024 surrogate regret among checkpoints within 2% of epoch-0 normalized prediction score; actual economics required separately.',metrics=state['metrics'],parameter_updates=False,other_seeds_started=False),indent=2))
    print('EXPORTED',epoch,'APIC',identity,flush=True)
    del model,base;torch.cuda.empty_cache()
if args.epochs is None:
    (ROOT/'results'/f'primary_checkpoint_candidates_s{args.seed}.json').write_text(json.dumps(dict(epochs=epochs,selection_year=2024,price_tolerance_fraction=.02),indent=2))
