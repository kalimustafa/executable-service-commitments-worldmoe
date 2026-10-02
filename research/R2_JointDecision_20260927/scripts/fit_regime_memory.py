"""Fit convex multiscale error-memory weights, with no evaluation-year fitting."""
import sys,json,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
from src.data.regime_memory import market_history,memory_features,export_correction
import torch,numpy as np
from scipy.optimize import minimize

torch.set_num_threads(2);data,mean,q=market_history();memory=memory_features(data,mean);weights=np.zeros((2,4,6));fits=[]
tr=data['train_indices'];va=data['selection_indices'];target=data['labels'][:,2];zone=data['zone'];delta=np.zeros_like(mean)
assert set(str(d)[:4] for d in data['dates'][tr])=={'2021','2022','2023'}
for gate in [0,1]:
    for site in range(4):
        ids=tr[zone[tr]==site];a=memory[gate,ids];y=target[ids]-mean[gate,ids];scale=max(np.std(y),1.)
        def objective(w):
            error=(a@w-y)/scale
            return float(np.mean(error**2)+.001*np.dot(w,w)),2*a.T@error/(len(a)*scale)+.002*w
        fit=minimize(objective,np.full(6,1/12),jac=True,method='SLSQP',bounds=[(0,1)]*6,constraints=[{'type':'ineq','fun':lambda w:1-w.sum(),'jac':lambda w:-np.ones(6)}],options={'ftol':1e-10,'maxiter':100})
        assert fit.success and fit.x.sum()<=1+1e-7
        weights[gate,site]=fit.x;fits.append(dict(gate=gate,site=site,iterations=int(fit.nit),zero_component=1-float(fit.x.sum())))
    delta[gate]=np.clip(np.einsum('ij,ij->i',memory[gate],weights[gate,zone]),-15,15)
baseline_error=mean[:,va]-target[va];updated=baseline_error+delta[:,va]
baseline_rmse=float(np.sqrt(np.mean(baseline_error**2)));new_rmse=float(np.sqrt(np.mean(updated**2)))
source=ROOT/'checkpoints/QPR2_seed20260920_epoch10.pt';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
checkpoint=ROOT/'checkpoints/RegimeR2_seed20260920.pt';torch.save(dict(weights=weights,seed=20260920,architecture='convex multiscale causal error-memory layer',forecast_epoch=10,forecast_checkpoint_sha256=source_hash,training_years=[2021,2022,2023],selection_year=2024,clip_usd_mwh=15.),checkpoint);checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
result=dict(seed=20260920,training_years=[2021,2022,2023],selection_year=2024,baseline_selection_rmse=baseline_rmse,selection_rmse=new_rmse,baseline_bias=float(baseline_error.mean()),selection_bias=float(updated.mean()),weights=weights.tolist(),fits=fits,forecast_guard_passed=new_rmse<=1.02*baseline_rmse,checkpoint_sha256=checkpoint_hash,feedback_rule='target local date minus two days; latest source day ends before the registered pre-DA publication gate',evaluation_year_weight_updates=False,other_seeds_started=False)
print('REGIME FIT',json.dumps(result),flush=True)
if result['forecast_guard_passed']:
    export_correction(data,memory,weights,2024,checkpoint_hash)
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta','0','--safety','0.9','--epoch','10','--regime'],check=True,cwd=ROOT)
    result['economics']=json.loads((ROOT/'results/RegimeR2_Eagle_2024_s20260920.json').read_text())
else:result['status']='rejected before annual replay by forecast-quality guard'
(ROOT/'results/regime_primary_complete.json').write_text(json.dumps(result,indent=2));print('REGIME COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['weights','fits','economics']}),flush=True)
