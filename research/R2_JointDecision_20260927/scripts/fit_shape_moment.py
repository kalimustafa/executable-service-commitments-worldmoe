"""Fit a 36-parameter distribution-completion layer; reject before replay if needed."""
import sys,json,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.runtime import ROOT
from src.models.coupled_rollout import ArchitectedR2QPHead
from src.models.shape_moment_r2 import ShapeMomentR2,shape_features
import torch,numpy as np
from scipy.optimize import minimize
from scipy.special import expit

torch.set_num_threads(2);source=ROOT/'checkpoints/QPR2_seed20260920_epoch10.pt';core=ArchitectedR2QPHead();core.load_state_dict(torch.load(source,map_location='cpu',weights_only=False)['state_dict']);core.eval()
with np.load(ROOT/'cache/rollout_forecast_features.npz') as z:
    n=len(z['zone']);zone=torch.tensor(np.tile(z['zone'],2));gate=torch.tensor(np.repeat(np.arange(2),n));truth=np.tile(z['labels'][:,2],2).astype(float)
    tr=np.r_[z['train_indices'],z['train_indices']+n];va=np.r_[z['selection_indices'],z['selection_indices']+n];qs=[]
    assert set(str(d)[:4] for d in z['dates'][z['train_indices']])=={'2021','2022','2023'}
    with torch.no_grad():
        for g in [0,1]:
            _,shift=core.market_update(torch.tensor(z[f'decision_latent_{g}']));qs.append(torch.tensor(z[f'rt_{g}'])+shift[:,None])
q=torch.cat(qs);raw=shape_features(q,zone,gate).numpy().astype(float);center=raw[tr].mean(0);scale=np.maximum(raw[tr].std(0),.05)
x=np.c_[np.ones(len(raw)),np.clip((raw-center)/scale,-8,8)];q=q.numpy().astype(float)
target=np.c_[np.maximum(truth-q[:,-1],0),np.maximum(q[:,0]-truth,0)];coefficients=np.zeros((3,12));fits=[]
for head in range(2):
    a=x[tr];y=target[tr,head];normalizer=max(y.mean(),.1);penalty=np.r_[0,np.full(11,.01)]
    def objective(beta):
        score=a@beta;mu=np.logaddexp(0,score).clip(1e-9);sigmoid=expit(score)
        value=np.mean(mu-y*np.log(mu))/normalizer+.5*np.dot(penalty,beta*beta)
        grad=a.T@((1-y/mu)*sigmoid)/(len(a)*normalizer)+penalty*beta
        return value,grad
    initial=np.zeros(12);initial[0]=np.log(np.expm1(max(y.mean(),.02)))
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',options={'maxiter':150,'ftol':1e-10,'gtol':1e-6})
    assert np.isfinite(result.fun);coefficients[head]=result.x;fits.append(dict(success=bool(result.success),iterations=int(result.nit),loss=float(result.fun)))
w=np.array([.075,.1,.2,.25,.2,.1,.075]);base=q@w;gaps=np.array([.05,.15,.25,.25,.15,.05]);width=np.sum(np.diff(q,axis=1)*gaps,1)
clipped=np.maximum(q[:,0],np.minimum(truth,q[:,-1]));central_scale=max(np.std(clipped[tr]-base[tr]),1.)
a=x[tr]*width[tr,None]/central_scale;b=(clipped[tr]-base[tr])/central_scale
coefficients[2]=np.linalg.solve(a.T@a/len(a)+np.diag(np.r_[1e-6,np.full(11,.01)]),a.T@b/len(a))
scores=x@coefficients.T;mu=np.logaddexp(0,scores[:,:2]);delta=mu[:,0]-mu[:,1]+width*np.clip(scores[:,2],-.5,.5)
rmse=float(np.sqrt(np.mean((base[va]-truth[va])**2)));new_rmse=float(np.sqrt(np.mean((base[va]+delta[va]-truth[va])**2)))
model=ShapeMomentR2()
with torch.no_grad():model.feature_center.copy_(torch.tensor(center));model.feature_scale.copy_(torch.tensor(scale));model.coefficients.copy_(torch.tensor(coefficients))
checkpoint=ROOT/'checkpoints/ShapeR2_seed20260920_epoch00.pt';source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
torch.save(dict(state_dict=model.state_dict(),seed=20260920,epoch=0,architecture='ShapeMomentR2',forecast_epoch=10,forecast_checkpoint_sha256=source_hash,training_years=[2021,2022,2023],selection_year=2024,parameters=36,tail_optimization=fits,selection_rmse=new_rmse,baseline_selection_rmse=rmse),checkpoint)
result=dict(seed=20260920,parameters=36,training_years=[2021,2022,2023],selection_year=2024,tail_optimization=fits,baseline_selection_rmse=rmse,selection_rmse=new_rmse,selection_bias=float(np.mean(base[va]+delta[va]-truth[va])),forecast_guard_passed=new_rmse<=1.02*rmse,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),other_seeds_started=False)
print('SHAPE FIT',json.dumps(result),flush=True)
if result['forecast_guard_passed']:
    subprocess.run([sys.executable,'-B','-u',str(ROOT/'scripts/evaluate_scenario_recourse.py'),'--eta','0','--safety','0.9','--epoch','10','--shape-epoch','0'],check=True,cwd=ROOT)
    result['economics']=json.loads((ROOT/'results/ShapeR2_Eagle_2024_s20260920_ep00.json').read_text())
else:result['status']='rejected before annual replay by the forecast-quality guard'
(ROOT/'results/shape_primary_complete.json').write_text(json.dumps(result,indent=2));print('SHAPE COMPLETE',json.dumps({k:v for k,v in result.items() if k!='economics'}),flush=True)
