from src.runtime import ROOT
from src.data.access_guard import install_guard
install_guard()
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from src.data.r2_data import R2Data
from src.calibration.calibrate_and_predict import load_model
from src.models.ordered_residual_trees import R2OrderedResidualTrees
from src.training.losses import price_losses
from src.training.checkpoint_selection import selection_score

def cache_rows(model,data,indices):
    store={k:[] for k in ['features','latent','raw','indices','publication']}
    with torch.inference_mode():
        for p in [0.,1.]:
            for ix in indices.split(4096):
                pub=torch.full((len(ix),),p,device=data.device);batch=data.batch(ix)
                out=model.source(batch,pub,data.labels[ix,0]);z=out['latent'];base=model.source.base
                vals=dict(features=model.features(batch,pub,data.labels[ix,0],out),latent=z,
                    raw=torch.cat([base.da_head(z),base.dart_head(z)],-1).float(),indices=ix,publication=pub)
                for k,v in vals.items():store[k].append(v.detach().cpu())
    return {k:torch.cat(v).numpy() for k,v in store.items()}

def pack_score(model,data,pack,prediction,gradient=False):
    total=0.;raw_loss=0.;grads=[]
    for start in range(0,len(prediction),4096):
        sl=slice(start,start+4096);ids=torch.as_tensor(pack['indices'][sl],device=data.device)
        latent=torch.tensor(pack['latent'][sl],device=data.device)
        raw=torch.tensor(pack['raw'][sl]+np.clip(prediction[sl],-1,1),device=data.device,requires_grad=gradient)
        pub=torch.tensor(pack['publication'][sl],device=data.device)
        out=model.decode(latent,raw,data.zone[ids],pub,data.labels[ids,0])
        _,score=price_losses(out,data.labels[ids],data.target_scale[data.zone[ids]])
        total+=float(score.detach().sum())
        if gradient:
            score.sum().backward();grads.append(raw.grad.detach().cpu().numpy())
    return total/len(prediction),np.concatenate(grads) if gradient else None

def main():
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.manual_seed(20260920)
    data=R2Data(multiscale=True);source,source_cp=load_model('C0_Repaired',20260920,data)
    model=R2OrderedResidualTrees(source).to(data.device).eval()
    train=data.indices['train'];unique=torch.unique(torch.stack([data.world_index[train],data.zone[train]],-1),dim=0)
    unique=unique[unique[:,0]>0];wi,zone=unique[:,0],unique[:,1]
    world=torch.cat([data.world['w0_local'][wi,zone],data.world['w1_local'][wi,zone],data.world['w1_global'][wi],data.world['tail_context'][wi]],-1).float().cpu().numpy()
    pca=PCA(n_components=16,svd_solver='randomized',random_state=20260920).fit(world)
    model.pca_center.copy_(torch.tensor(pca.mean_,device=data.device));model.pca_components.copy_(torch.tensor(pca.components_,device=data.device))
    tr=cache_rows(model,data,train);val=cache_rows(model,data,data.indices['selection'])
    prediction=np.zeros_like(tr['raw']);validation=np.zeros_like(val['raw']);best,_=pack_score(model,data,val,validation)
    baseline_score=best;best_count=0;logs=[];started=time.perf_counter();rng=np.random.default_rng(20260920)
    def save(count,score):
        torch.save(dict(schema='r2.ordered_residual_trees.v1',model='C2_OrderedResidualTrees',seed=20260920,epoch=count,stage='residual_trees',
            state_dict={k:v.cpu() for k,v in model.state_dict().items()},trees=model.trees.copy(),learning_rate=model.learning_rate,
            source_checkpoint='C0_Repaired_seed20260920_best.ckpt',validation=dict(price_score=score),split_hash=data.split_hash,
            normalization=data.normalization,selection_precision='FP32',w1_trainable=0),ROOT/'checkpoints/C2_OrderedResidualTrees_seed20260920_best.ckpt')
    save(0,best);print('TREE ZERO SCORE',best,flush=True)
    for iteration in range(1,81):
        train_score,gradient=pack_score(model,data,tr,prediction,gradient=True)
        sample=rng.choice(len(prediction),size=len(prediction)//2,replace=False)
        tree=DecisionTreeRegressor(max_depth=4,min_samples_leaf=256,random_state=20260920+iteration)
        tree.fit(tr['features'][sample],-gradient[sample])
        model.trees.append(tree)
        prediction+=model.learning_rate*tree.predict(tr['features']).astype(np.float32)
        validation+=model.learning_rate*tree.predict(val['features']).astype(np.float32)
        if iteration==1 or iteration%5==0:
            score,_=pack_score(model,data,val,validation)
            row=dict(model='C2_OrderedResidualTrees',seed=20260920,trees=iteration,train_score=train_score,selection_price_score=score,seconds=time.perf_counter()-started)
            logs.append(row)
            if score<best-1e-6:best=score;best_count=iteration;save(iteration,score)
            pd.DataFrame(logs).to_csv(ROOT/'logs/ordered_residual_trees_training.csv',index=False)
            print(json.dumps(dict(**row,best_trees=best_count)),flush=True)
    (ROOT/'logs/C2_OrderedResidualTrees_seed20260920_completed.json').write_text(json.dumps(dict(model='C2_OrderedResidualTrees',seed=20260920,
        best_epoch=best_count,best_price_score=best,initial_price_score=baseline_score,epochs_run=80,seconds=time.perf_counter()-started,
        split_hash=data.split_hash,source_frozen=True,world_pca_fit='unique 2022-2023 training contexts only',status='complete'),indent=2))

if __name__=='__main__':main()
