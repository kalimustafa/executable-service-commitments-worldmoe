from src.models.r2_wgr import R2WGR

class R2DeepNoGate(R2WGR):
    def __init__(self,base):
        super().__init__(base,gated=False,world_dropout=0.)

