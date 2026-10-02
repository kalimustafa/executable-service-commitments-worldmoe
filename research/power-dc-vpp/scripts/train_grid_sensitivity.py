"""Fixed-budget train-only short R-B adaptations; no 2025 input or selection."""
from __future__ import annotations
import argparse
import gc
import hashlib
import _sha256
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import pyarrow.dataset  # noqa: F401 -- required verified Windows import order
import numpy as np
import torch
import run_matched_mpc_benchmark as common
from multiseed_common import write_json, read_json, now
from dcvpp.vpp_world_moe.r_ecmoe import RiskConditionPacket
from dcvpp.vpp_world_moe.r_ecmoe_losses import r_ecmoe_loss, RECMoELossWeights
from dcvpp.vpp_world_moe.r_ecmoe_replay import RealizedHistoricalReplay
from dcvpp.vpp_world_moe.grid_externality_loss import grid_externality
from dcvpp.vpp_world_moe.data import _slot_feature_statistic, _normalization_center

ROOT = REPO / 'artifacts/r_ecmoe_closedloop_sensitivity_20260913'
WEIGHTS = (0., .1, 1., 10.)
ORIGINS = (0, 32, 48, 64)
SEED = 20240813


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, _sha256.sha256).hexdigest()


def reset_rng():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


def previous_grids(batch, train_records, origin):
    """Loss-side completed physical grids, not issue-baseline fast-slot proxies.

    State row t stores the completed action at t-1. No current/future physical
    row enters this boundary. Origin0 has the explicitly reset zero boundary.
    """
    packet=batch['world_input'];device=packet.fast_values.device
    if origin==0:
        z=torch.zeros(packet.fast_values.shape[0],4,device=device)
        return z,z.clone()
    if origin<2:raise ValueError('Two completed grid observations are required')
    previous=[];previous_previous=[]
    for day in batch['operating_dates']:
        states=train_records[str(day)]['base_record']['state_sequence']
        previous.append(torch.as_tensor(states[origin,12:16]))
        previous_previous.append(torch.as_tensor(states[origin-1,12:16]))
    return torch.stack(previous).to(device),torch.stack(previous_previous).to(device)


def named_weights():
    snap = read_json(REPO / 'artifacts/r_ecmoe_mechanisms_2025_20260910/epoch130-seed20240813-20260910T204056/launch_inputs.json')
    loss = snap['main_config_resolved']['loss']
    mapping = dict(expected_operating_cost='expected_cost', upper_tail_cost='tail_cost',
        quantile_pinball='quantile', sla_deadline='sla_deadline', constraint_residual='constraint',
        route_continuation='route_continuation', load_balance='load_balance', tail_risk='tail_expert',
        action_smoothness='smoothness', safe_anchor='behavior_anchor')
    return RECMoELossWeights(**{dst: loss[src] for dst, src in mapping.items()})


def run(root, device, pilot=False):
    root.mkdir(parents=True, exist_ok=True)
    record, wc, mc = common.input_record('legacy_seed20240813')
    corpus = common.WorldMoECorpus(wc, REPO)
    train_records={str(r['operating_date']):r for r in corpus.records('train',require_h96=False)}
    weights = named_weights()
    protocol = dict(schema='dcvpp.grid.short-adaptation.v1', fixed_at=now(),
        source_record=record, training_years=[2022,2023], parameters_fit_on_2025=False,
        training_seed=SEED, lambda_grid=list(WEIGHTS), epochs_per_arm=4,
        epoch_origins=list(ORIGINS), horizon=16, batch_size=64, learning_rate=.0001,
        adamw_betas=[.9,.95], weight_decay=.01, gradient_clip_norm=1., precision='bf16',
        original_loss_weights=weights.as_mapping(), safe_anchor_progress=1.,
        arm_initialization='same frozen Run1 R130, with fresh equal AdamW state and data order',
        selection='fixed final epoch4; no validation or test-based arm/checkpoint selection',
        scope='exploratory one-lineage short continuation, not independent retraining repetitions',
        penalty='mean_tz abs(delta signed grid)/local import+export range + mean_t positive second difference of total signed grid/total import+export range',
        penalty_outside_cost_cvar=True, world_model_frozen=True,
        boundary_grid_source='completed physical state rows origin and origin-1 at RT origins32/48/64; explicit reset zero boundary at origin0; not issue-baseline grid proxies',
        input_sha256={str(p):sha(p) for p in [Path(record['checkpoint']),Path(record['w1_checkpoint']),Path(__file__),REPO/'src/dcvpp/vpp_world_moe/grid_externality_loss.py']})
    if not pilot:
        prior = read_json(root/'training_protocol.json')
        if prior:
            comparison=dict(prior); comparison.pop('fixed_at'); candidate=dict(protocol); candidate.pop('fixed_at')
            if comparison != candidate: raise RuntimeError('Existing frozen training protocol differs')
            protocol=prior
        else: write_json(root/'training_protocol.json',protocol)
    for lam in ((1.,) if pilot else WEIGHTS):
        tag=f'lambda_{lam:g}'; folder=root/('pilot_train' if pilot else 'adaptations')/tag
        folder.mkdir(parents=True,exist_ok=True)
        if not pilot and read_json(folder/'completed.json'): continue
        model=common.runtime(record,wc,mc,device)
        for name,param in model.named_parameters(): param.requires_grad_(not name.startswith('world_model.'))
        model.configure_stage('rb');model.train()
        assert not any(p.requires_grad for p in model.world_model.parameters())
        initial_r={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('world_model.')}
        world_versions={name:p._version for name,p in model.world_model.named_parameters()}
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=.0001,betas=(.9,.95),weight_decay=.01,fused=device.type=='cuda')
        replay=RealizedHistoricalReplay.from_model(model); rolling=None; updates=0; started=time.perf_counter()
        reset_rng()
        for epoch,origin in enumerate(ORIGINS,1):
            for batch_id,batch in enumerate(corpus.batches('train',2 if pilot else 64,device=device,
                    shuffle_seed=SEED+epoch,max_days=2 if pilot else None,origin=origin,
                    require_h96=False,include_next_day=False),1):
                if any(str(d)[:4] not in {'2022','2023'} for d in batch['operating_dates']): raise RuntimeError('Non-training year')
                packet=batch['world_input']; b=packet.fast_values.shape[0]
                risk=RiskConditionPacket.defaults(b,device=device,horizon=16,stage=1,action_range=1.)
                optimizer.zero_grad(set_to_none=True)
                with common.core.base._precision(device):
                    decision=model(packet,risk,da_anchor=batch['forecast_anchors']['da'],da_residual_cap=batch['forecast_residual_caps']['da'],return_diagnostics=False)
                output=replay(decision,batch=batch,risk=risk,source_split='train')
                loss=r_ecmoe_loss(output,cost_scale=batch['train_frozen_baseline_daily_cost']*(16/96),cost_scale_source='train',progress=1.,weights=weights,rolling_route_load=rolling)
                previous,previous_previous=previous_grids(batch,train_records,origin)
                c=packet.constraints
                penalty,ramp,curvature=grid_externality(output.actions.grid[:,origin:origin+16],previous,previous_previous,c.grid_import_limit+c.grid_export_limit)
                total=loss.total+lam*penalty.mean()
                if not torch.isfinite(total): raise RuntimeError('Nonfinite training loss')
                total.backward(); norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradient')
                optimizer.step();rolling=loss.next_rolling_route_load.detach();updates+=1
                row=dict(epoch=epoch,batch=batch_id,origin=origin,updates=updates,lambda_grid=lam,
                    loss=float(total.detach()),base_loss=float(loss.total.detach()),grid_penalty=float(penalty.mean().detach()),
                    local_ramp=float(ramp.mean().detach()),positive_curvature=float(curvature.mean().detach()),
                    gradient_norm=float(norm),elapsed_seconds=time.perf_counter()-started,
                    dates=list(batch['operating_dates']),updated_at=now())
                with (folder/'train.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                write_json(root/'training_progress.json',dict(status='pilot' if pilot else 'training',arm=tag,**row))
                print(json.dumps({k:v for k,v in row.items() if k!='dates'}),flush=True)
            if pilot:break
        assert all(p._version==world_versions[name] for name,p in model.world_model.named_parameters())
        changed=sum(not torch.equal(initial_r[k],v.detach().cpu()) for k,v in model.state_dict().items() if k in initial_r)
        assert changed>0
        if not pilot:
            checkpoint=folder/'fixed_epoch4_r.pt'
            torch.save(dict(schema='dcvpp.short-r-adaptation.v1',r_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('world_model.')},
                optimizer=optimizer.state_dict(),lambda_grid=lam,epochs=4,updates=updates,source_record=record,training_protocol_sha256=sha(root/'training_protocol.json')),
                checkpoint)
            write_json(folder/'completed.json',dict(status='completed',checkpoint=str(checkpoint),sha256=sha(checkpoint),updates=updates,epochs=4,elapsed_seconds=time.perf_counter()-started,
                world_model_unchanged=True,changed_r_tensors=changed,completed_at=now()))
        else:write_json(folder/'result.json',dict(status='passed',changed_r_tensors=changed,world_model_unchanged=True,updates=updates))
        del model,optimizer,params,initial_r,output,decision,loss,total;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    if not pilot:write_json(root/'training_completed.json',dict(status='completed',arms=4,epochs_each=4,completed_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    run(a.root,torch.device(a.device),a.pilot)
