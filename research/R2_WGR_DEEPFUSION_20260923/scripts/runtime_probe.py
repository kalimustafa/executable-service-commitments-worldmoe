from src.runtime import *
from src.data.access_guard import install_guard
install_guard()
import torch
torch.set_num_threads(4)
print('torch', torch.__version__, flush=True)
import psutil
def dlls():
    print([x.path for x in psutil.Process().memory_maps() if any(s in x.path.lower() for s in ['iomp','libomp','mkl','openblas'])], flush=True)
dlls()
print('clone before', torch.rand(100000).clone().shape, flush=True)
from dcvpp.vpp_world_moe.config import WorldMoEConfig
from dcvpp.vpp_world_moe.model import VPPWorldMoE
dlls()
print('clone after', torch.rand(100000).clone().shape, flush=True)
from dcvpp.vpp_world_moe.config import _read_config_with_base
cfg = _read_config_with_base(REPO / 'configs/multiseed_20260910_20240829_w1.yaml')
print('config read', flush=True)
model = VPPWorldMoE(WorldMoEConfig.from_mapping(cfg['model']), include_legacy_planner=False)
print('model created', sum(p.numel() for p in model.parameters()), flush=True)
