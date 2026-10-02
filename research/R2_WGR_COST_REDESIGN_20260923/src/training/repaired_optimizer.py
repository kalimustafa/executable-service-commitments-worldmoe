import torch

def warmup_optimizer(model):
    model.base.requires_grad_(False);model.base.eval()
    base_ids={id(p) for p in model.base.parameters()}
    fusion=[p for p in model.parameters() if id(p) not in base_ids]
    assert all(p.requires_grad for p in fusion)
    return torch.optim.AdamW([dict(params=fusion,lr=.001,name='fusion')],weight_decay=1e-4)

def restore_best_for_joint(model,optimizer,warmup_payload):
    model.load_state_dict(warmup_payload['state_dict'],strict=True)
    optimizer.load_state_dict(warmup_payload['optimizer_state'])
    restored_steps=[float(v['step']) for v in optimizer.state.values() if 'step' in v]
    encoder=[];heads=[]
    for name,part in model.base.named_children():
        if name in {'numerical','shared'}:
            part.requires_grad_(True);encoder.extend(part.parameters())
        elif name in {'da_head','dart_head','dependence','regime_head'}:
            part.requires_grad_(True);heads.extend(part.parameters())
        else:part.requires_grad_(False)
    optimizer.param_groups[0]['lr']=.0003
    optimizer.add_param_group(dict(params=encoder,lr=.00002,name='base_encoder'))
    optimizer.add_param_group(dict(params=heads,lr=.00005,name='base_heads'))
    after_steps=[float(v['step']) for v in optimizer.state.values() if 'step' in v]
    assert restored_steps==after_steps
    model.base.eval()
    return dict(restored_epoch=warmup_payload['epoch'],optimizer_moments_preserved=True,
                restored_fusion_states=len(restored_steps),max_restored_step=max(restored_steps,default=0.),
                base_encoder_lr=.00002,base_heads_lr=.00005,fusion_lr=.0003)
