"""Confirm the original 2025 savings target using one frozen primary seed only."""
import sys,argparse,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT,WORKSPACE
import torch,numpy as np,pandas as pd
from src.data.structured_final_data import StructuredFinalData
from src.data.r2_data import ZONES
from src.models.factory import make_model
from src.models.structured_decision import StructuredDecision
from src.models.coupled_rollout import ArchitectedR2Head,ArchitectedR2QPHead
from src.models.distributional_r2 import DistributionalR2Head

p=argparse.ArgumentParser();p.add_argument('--family',choices=['DeepR2','QPR2','DistR2'],default='QPR2');args=p.parse_args();family=args.family;seed=20260920;label={'DeepR2':'deep','QPR2':'qp','DistR2':'distribution'}[family]
summary=json.loads((ROOT/'results'/f'{label}_primary_complete.json').read_text());assert summary['status']=='complete' and summary['best_restored']
epoch=summary['best_epoch'];checkpoint=ROOT/'checkpoints'/f'{family}_seed{seed}_epoch{epoch:02d}.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
freeze=dict(family=family,seed=seed,selected_epoch=epoch,selection_year=2024,selection_savings_usd=summary['best_2024_savings_usd'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),frozen_utc=datetime.now(timezone.utc).isoformat(),evaluation_year=2025,evaluation_role='exploratory frozen confirmation on a previously examined calendar',target_savings_usd=106084.33667311305,extra_seeds_permitted_before_success=False)
(ROOT/'results'/f'primary_freeze_{family}.json').write_text(json.dumps(freeze,indent=2))
torch.set_num_threads(3);data=StructuredFinalData(seed)
base=StructuredDecision(make_model('C1_ProfileResidual',seed,data),data.target_scale,data.target_center[:,2],torch.ones(4,device='cuda')).cuda()
base.load_state_dict(torch.load(ROOT/'checkpoints/StructuredR2_seed20260920_epoch10.pt',map_location='cpu',weights_only=False)['state_dict']);base.eval().requires_grad_(False)
head=({'DeepR2':ArchitectedR2Head,'QPR2':ArchitectedR2QPHead,'DistR2':DistributionalR2Head}[family])().cuda();head.load_state_dict(payload['state_dict']);head.eval().requires_grad_(False)
with np.load(WORKSPACE/'external_validation/data/forcing_2025.npz') as z:times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
hours=times.floor('h').unique();index=pd.MultiIndex.from_product([hours,ZONES],names=['timestamp_utc','zone']);frame=data.frame.copy();frame.timestamp_utc=pd.to_datetime(frame.timestamp_utc,utc=True)
lookup=pd.Series(np.arange(len(frame)),index=pd.MultiIndex.from_frame(frame[['timestamp_utc','zone']]))
indices=torch.tensor(lookup.reindex(index).to_numpy(np.int64),device='cuda');arrays=dict(hours=hours.asi8,pue_reference=base.pue_reference.cpu().numpy(),historical_rt=base.historical_rt.cpu().numpy());apic=0.;price_rows=[]
tau=torch.tensor([.05,.1,.25,.5,.75,.9,.95],device='cuda')
with torch.inference_mode():
    for gate in [0,1]:
        prices=[];latent=[]
        for ids in indices.split(4096):
            y=data.labels[ids]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=base(data.batch(ids),torch.full((len(ids),),float(gate),device='cuda'),y[:,0]);_,shift=head.market_update(out['decision_latent'])
            rt=out['rt']+shift[:,None];dart=out['dart']-shift[:,None]
            assert torch.diff(rt,dim=-1).min()>=-1e-5
            if gate:apic=max(apic,float((rt-(y[:,0,None]-dart.flip(-1))).abs().max()))
            terms=[]
            for j,q in enumerate([out['da'],dart,rt]):
                error=y[:,j,None]-q;terms.append(torch.maximum(tau*error,(tau-1)*error).mean(-1)/data.target_scale[data.zone[ids],j])
            price_rows.append((.5*terms[0]+terms[1]+.5*terms[2]).cpu())
            prices.append(out['rt'].float().cpu().numpy());latent.append(out['decision_latent'].float().cpu().numpy())
        arrays[f'rt_{gate}']=np.concatenate(prices).reshape(len(hours),4,7);arrays[f'latent_{gate}']=np.concatenate(latent).reshape(len(hours),4,128)
np.savez_compressed(ROOT/'cache/structured_market2025_s20260920_ep10.npz',**arrays)
(ROOT/'results'/f'primary_forecast_2025_{family}.json').write_text(json.dumps(dict(family=family,seed=seed,epoch=epoch,price_score=float(torch.cat(price_rows).mean()),post_DA_APIC_max_error=apic,base_cache_note='Frozen R2 forecasts; the stage head applies its learned APIC-preserving shift at execution',parameter_updates=False),indent=2))
del base,head,data;torch.cuda.empty_cache()
command=[sys.executable,'-B','-u',str(ROOT/'scripts/replay_development.py'),'--deep','--structured','--epochs',str(epoch),'--solar-hours','6','--year','2025']
if family in ['QPR2','DistR2']:command+=['--qp',str(payload['qp_regularization'])]
if family=='DistR2':command+=['--distribution']
subprocess.run(command,check=True,cwd=ROOT)
row=json.loads((ROOT/'results'/f'{family}_Eagle_2025_s20260920_ep{epoch:02d}.json').read_text())
passed=row['saving_usd']>=106084.33667311305 and row['late_IT_mwh']<1e-8 and row['closing_it_mwh']<1e-8
gate=dict(passed=passed,primary_seed=seed,family=family,epoch=epoch,target_year=2025,target_savings_usd=106084.33667311305,measured_savings_usd=row['saving_usd'],capture_fraction=row['capture_lower_fraction'],late_it_mwh=row['late_IT_mwh'],closing_it_mwh=row['closing_it_mwh'],selection_year=2024,other_seeds_authorized=passed,evaluation_role='exploratory frozen confirmation')
(ROOT/'results/primary_80_gate.json').write_text(json.dumps(gate,indent=2));print('PRIMARY 80 GATE',json.dumps(gate),flush=True)
