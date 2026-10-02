import torch
from vendor.r2_market import R2Config
from src.models.r2_base import R2Base
from src.models.r2_shallow import R2Shallow
from src.models.r2_wgr import R2WGR
from src.models.r2_deep_nogate import R2DeepNoGate

def make_model(name,seed,data,base_checkpoint=None):
    scales=dict(da_center=data.target_center[:,0],da_scale=data.target_scale[:,0],dart_center=data.target_center[:,1],dart_scale=data.target_scale[:,1])
    config=R2Config(numerical_features=len(data.names),use_world=name=='A1_Shallow',seed=seed)
    base=(R2Shallow if name=='A1_Shallow' else R2Base)(config,**scales)
    if name in {'A0_Base','A1_Shallow'}:return base.to(data.device)
    if base_checkpoint is not None:
        payload=torch.load(base_checkpoint,map_location='cpu',weights_only=False)
        assert payload['model']=='A0_Base' and payload['seed']==seed
        base.load_state_dict(payload['state_dict'],strict=True)
    model=R2DeepNoGate(base) if name=='A2_Deep_NoGate' else R2WGR(base)
    return model.to(data.device)

