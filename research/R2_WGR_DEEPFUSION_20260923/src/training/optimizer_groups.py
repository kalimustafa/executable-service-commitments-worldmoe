import torch

def optimizer_for(model,stage):
    deep=hasattr(model,'base')
    if not deep:return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.002,weight_decay=1e-4)
    model.base.requires_grad_(True);model.base.slot_head.requires_grad_(False)
    if stage=='warmup':
        for part in ['numerical','shared','world_adapter','expert_key','expert_identity','expert_query']:
            getattr(model.base,part).requires_grad_(False)
    base=[p for p in model.base.parameters() if p.requires_grad]
    base_ids={id(p) for p in model.base.parameters()}
    fusion=[p for p in model.parameters() if p.requires_grad and id(p) not in base_ids]
    optimizer=torch.optim.AdamW([dict(params=base,lr=.002 if stage=='warmup' else .0003),dict(params=fusion,lr=.002)],weight_decay=1e-4)
    assert len({id(p) for g in optimizer.param_groups for p in g['params']})==len(base)+len(fusion)
    assert all('backbone' not in name and 'tokenizer' not in name for name,_ in model.named_parameters())
    return optimizer

