import sys,argparse,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT,WORKSPACE
import torch,numpy as np,pandas as pd
from src.data.structured_final_data import StructuredFinalData
from src.data.r2_data import ZONES
from src.models.factory import make_model
from src.models.structured_decision import StructuredDecision

p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[20260920]);args=p.parse_args();torch.set_num_threads(3)
for seed in args.seeds:
    selection=json.loads((ROOT/'results'/f'structured_complete_s{seed}.json').read_text());epoch=selection['best_epoch']
    data=StructuredFinalData(seed);base=make_model('C1_ProfileResidual',seed,data)
    model=StructuredDecision(base,data.target_scale,data.target_center[:,2],torch.ones(4,device='cuda')).cuda()
    state=torch.load(ROOT/'checkpoints'/f'StructuredR2_seed{seed}_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);model.eval().requires_grad_(False)
    with np.load(WORKSPACE/'external_validation/data/forcing_2025.npz') as z:times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True))
    hours=times.floor('h').unique();idx=pd.MultiIndex.from_product([hours,ZONES],names=['timestamp_utc','zone'])
    frame=data.frame.copy();frame.timestamp_utc=pd.to_datetime(frame.timestamp_utc,utc=True);lookup=pd.Series(np.arange(len(frame)),index=pd.MultiIndex.from_frame(frame[['timestamp_utc','zone']]))
    indices=torch.tensor(lookup.reindex(idx).to_numpy(np.int64),device='cuda');arrays=dict(hours=hours.asi8,pue_reference=model.pue_reference.cpu().numpy(),historical_rt=model.historical_rt.cpu().numpy());identity=0.
    with torch.inference_mode():
        for gate in [0,1]:
            q=[];lat=[]
            for ids in indices.split(4096):
                y=data.labels[ids]
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(data.batch(ids),torch.full((len(ids),),float(gate),device='cuda'),y[:,0])
                assert torch.diff(out['rt'],dim=-1).min()>=-1e-5
                if gate:identity=max(identity,float((out['rt']-(y[:,0,None]-out['dart'].flip(-1))).abs().max()))
                q.append(out['rt'].float().cpu().numpy());lat.append(out['decision_latent'].float().cpu().numpy())
            arrays[f'rt_{gate}']=np.concatenate(q).reshape(len(hours),4,7);arrays[f'latent_{gate}']=np.concatenate(lat).reshape(len(hours),4,128)
    output=ROOT/'cache'/f'structured_market2025_s{seed}_ep{epoch:02d}.npz';np.savez_compressed(output,**arrays)
    output.with_suffix('.json').write_text(json.dumps(dict(seed=seed,selected_epoch=epoch,apic_max_error=identity,forecast_rows=len(indices)*2,year=2025,parameter_updates=False,checkpoint_selected_on=2024),indent=2))
    del model,base,data;torch.cuda.empty_cache()
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/replay_development.py'),'--structured','--seed',str(seed),'--epochs',str(epoch),'--year','2025'],check=True,cwd=ROOT)
