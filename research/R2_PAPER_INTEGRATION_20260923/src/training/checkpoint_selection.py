import torch
from vendor.r2_market import pinball
from src.training.losses import price_losses

def selection_score(model,data,batch_size=4096):
    model.eval();sums=torch.zeros(5,device=data.device);count=0
    with torch.inference_mode():
        for state in [0.,1.]:
            for ix in data.indices['selection'].split(batch_size):
                y,_=data.row_targets(ix);pub=torch.full((len(ix),),state,device=data.device)
                with torch.autocast('cuda',enabled=False):out=model(data.batch(ix),pub,y[:,0])
                losses,score=price_losses(out,y,data.target_scale[data.zone[ix]])
                raw=.5*pinball(out['da'],y[:,0])+pinball(out['dart'],y[:,1])+.5*pinball(out['rt'],y[:,2])
                sums+=torch.stack([x.sum() for x in losses]+[score.sum(),raw.sum()]);count+=len(ix)
    a=(sums/count).cpu().tolist()
    return dict(da_pinball_normalized=a[0],dart_pinball_normalized=a[1],rt_pinball_normalized=a[2],price_score=a[3],raw_weighted_pinball=a[4])
