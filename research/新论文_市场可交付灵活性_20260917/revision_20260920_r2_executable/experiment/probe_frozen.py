"""Strict selected-checkpoint load and one development input-contract probe."""
from pathlib import Path
import os,sys,json,time,gc
os.environ.update(PYTHONUTF8='1',CUDA_LAUNCH_BLOCKING='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]/'power-dc-vpp';sys.path.insert(0,str(REPO/'src'))
import pyarrow.dataset
import torch
from dcvpp.vpp_world_moe.config import _read_config_with_base,WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE,load_plannerless_released_w1_state
from dcvpp.vpp_world_moe.data import WorldMoECorpus

def main():
    start=time.perf_counter();torch.set_num_threads(4)
    s=json.loads((ROOT/'selected_w0_w1.json').read_text(encoding='utf-8'))
    cfg=_read_config_with_base(Path(s['config']));mc=WorldMoEConfig.from_mapping(cfg['model'])
    model=VPPWorldMoE(mc,include_legacy_planner=False)
    loads={}
    for phase in ['w0','w1']:
        ck=torch.load(s[phase]['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        assert ck['phase']==phase and ck['locked_2025_opened'] is False
        load_plannerless_released_w1_state(model,ck['model_state'])
        loads[phase]=dict(epoch=ck['resume_state']['epoch'],strict_load=True)
        del ck;gc.collect()
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    params=dict(model.named_parameters());groups=model.parameter_groups()
    w0names=set(name for group in ['encoder','shared','router'] for name in groups[group])
    counts=dict(total_parameters=sum(p.numel() for p in params.values()),
        frozen_W0_parameters=sum(params[n].numel() for n in w0names),
        frozen_W1_parameters=sum(p.numel() for n,p in params.items() if n not in w0names),
        trainable_parameters_W0=sum(params[n].numel() for n in w0names if params[n].requires_grad),
        trainable_parameters_W1=sum(p.numel() for n,p in params.items() if n not in w0names and p.requires_grad),
        trainable_R2_parameters=None,checkpoint_loads=loads,
        partition='W0-trained encoder/shared/router groups and remaining W1 groups in the final W1 snapshot; disjoint counts, no duplicate encoder')
    (ROOT/'frozen_parameter_report.json').write_text(json.dumps(counts,indent=2))
    print('FROZEN',json.dumps(counts),flush=True)
    del model;gc.collect()
    corpus=WorldMoECorpus(cfg,REPO,materialize=False)
    print('CORPUS_LOADED seconds',time.perf_counter()-start,flush=True)
    records=corpus.records('train',require_h96=False)
    record=records[20];print('RECORD_KEYS',list(record),flush=True)
    print('RECORD_TIME',record.get('operating_date'),record['base_record'].get('provenance'),flush=True)
    packet=corpus._packet([record],origin=19,device=torch.device('cpu'))
    from dataclasses import fields
    print('PACKET',[(f.name,tuple(getattr(packet,f.name).shape)) for f in fields(packet) if isinstance(getattr(packet,f.name),torch.Tensor)],flush=True)
    print('COUNTS',len(corpus.records('train')),len(corpus.records('validation')),flush=True)
    (ROOT/'results/input_probe.json').write_text(json.dumps(dict(status='complete',seconds=time.perf_counter()-start,
        record_keys=list(record),source_record_date=str(record.get('operating_date')),
        source_provenance=record['base_record'].get('provenance'),normalization_schema=corpus.statistics.schema if hasattr(corpus.statistics,'schema') else None),indent=2,default=str))
if __name__=='__main__':main()
