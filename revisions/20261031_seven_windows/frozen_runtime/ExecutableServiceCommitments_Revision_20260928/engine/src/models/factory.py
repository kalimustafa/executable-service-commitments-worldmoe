import torch
from vendor.r2_market import R2Config
from src.models.r2_base import R2Base
from src.models.r2_shallow import R2Shallow
from src.models.r2_repaired import R2WGR
from src.models.profile_residual import R2ProfileResidual
from src.models.temporal_world import R2TemporalWorld

def make_model(name,seed,data,base_checkpoint=None):
    scales=dict(da_center=data.target_center[:,0],da_scale=data.target_scale[:,0],dart_center=data.target_center[:,1],dart_scale=data.target_scale[:,1])
    config=R2Config(numerical_features=len(data.names),use_world=name=='A1_Shallow',seed=seed)
    base=(R2Shallow if name=='A1_Shallow' else R2Base)(config,**scales)
    if name in {'A0_Base','A1_Shallow'}:return base.to(data.device)
    if base_checkpoint is not None:
        payload=torch.load(base_checkpoint,map_location='cpu',weights_only=False)
        assert payload['model']=='A0_Base' and payload['seed']==seed
        base.load_state_dict(payload['state_dict'],strict=True)
    if name=='C1_ProfileResidual':model=R2ProfileResidual(base)
    elif name in {'C0_Repaired','C4_RecentReplay'}:model=R2WGR(base)
    elif name in {'C3_TemporalWorld','C5_TemporalReplay'}:model=R2TemporalWorld(base)
    else:raise ValueError('Failed-run architectures must not be trained here: '+name)
    return model.to(data.device)
