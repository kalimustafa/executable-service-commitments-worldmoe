import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import math

def slot_cost(x,rt,pue,pv,idle,k,refgrid):
    load=pue*(idle+k*x);grid=max(load-pv,0.);spill=max(pv-load,0.)
    return c.DT*(rt*grid+spill+c.FEE*abs(grid-refgrid))

def lower_hull(points):
    hull=[]
    for point in points:
        while len(hull)>=2:
            x0,y0=hull[-2];x1,y1=hull[-1];x2,y2=point
            if (y1-y0)*(x2-x1)>=(y2-y1)*(x1-x0):hull.pop()
            else:break
        hull.append(point)
    return hull

def zone_bound(z,arr,refgrid,window,prefix):
    n=c.N;slopes=[];width=[];slots=[];constant=[];convexified=0
    for t in range(n):
        cap=min(1.,max(0.,((c.c['grid_import_limit'][z]+c.pv[t,z])/c.pue[t,z]-c.idle[z])/c.k[z]))
        xs=sorted(set([0.,cap]+[float(x) for x in [(c.pv[t,z]/c.pue[t,z]-c.idle[z])/c.k[z],((c.pv[t,z]+refgrid[t,z])/c.pue[t,z]-c.idle[z])/c.k[z]] if 0<x<cap]))
        points=[(x,slot_cost(x,c.rt[t,z],c.pue[t,z],c.pv[t,z],c.idle[z],c.k[z],refgrid[t,z])) for x in xs]
        hull=lower_hull(points);convexified+=len(hull)<len(points);constant.append(points[0][1])
        for (x0,y0),(x1,y1) in zip(hull[:-1],hull[1:]):slopes.append((y1-y0)/(x1-x0));width.append(x1-x0);slots.append(t)
    cumulative=np.cumsum(arr[:,z]);lag=int(window*4)-1
    expired=np.r_[np.zeros(lag),cumulative[:-lag]];queue_upper=np.maximum(cumulative-expired,0.)
    m=len(slopes);obj=np.r_[slopes,np.zeros(n)];obj[-1]=c.DT*c.k[z]*max(c.c['pue_max'])*c.train_rt[z]
    upper=np.r_[width,queue_upper]
    matrix=coo_matrix((np.r_[np.ones(m+n),-np.ones(n-1)],(np.r_[slots,np.arange(n),np.arange(1,n)],np.r_[np.arange(m),m+np.arange(n),m+np.arange(n-1)])),shape=(n,m+n)).tocsr()
    result=linprog(obj,A_eq=matrix,b_eq=arr[:,z],bounds=list(zip(np.zeros_like(upper),upper)),method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not result.success:raise RuntimeError(result.message)
    pi=result.eqlin.marginals;reduced=obj-matrix.T@pi
    dual=float(pi@arr[:,z]+np.minimum(reduced,0)@upper+np.sum(constant));primal=float(result.fun+np.sum(constant))
    schedule=np.bincount(np.asarray(slots),weights=result.x[:m],minlength=n)
    true=float(sum(slot_cost(schedule[t],c.rt[t,z],c.pue[t,z],c.pv[t,z],c.idle[z],c.k[z],refgrid[t,z]) for t in range(n))+obj[-1]*result.x[-1])
    residual=float(np.max(np.abs(matrix@result.x-arr[:,z])))
    if residual>1e-6 or primal<dual-1e-5 or true<primal-1e-5:raise RuntimeError('Invalid bound certificate')
    np.savez_compressed(RESULTS/f'{prefix}_zone{z}_certificate.npz',multipliers=pi,reduced_cost=reduced,upper=upper,objective=obj,arrivals=arr[:,z],constant=constant,slot=slots,segment_width=width,relaxed_schedule=schedule,queue=result.x[m:])
    return dict(zone=z,dual_usd=dual,conservative_usd=math.floor(dual)-1,primal_usd=primal,true_relaxed_schedule_cost_usd=true,duality_gap_usd=primal-dual,balance_residual=residual,convexified_slots=convexified)

rows=[];zones=[]
for year in [2024,2025]:
    for trace in ['Eagle','Frontier']:
        arr,reference,shared=configure(year,trace);refcash,refgrid=c.cash(reference)
        refcost=float(refcash.sum()+shared.sum()+c.DT*np.sum(reference[c.N:]*c.k*max(c.c['pue_max'])*c.train_rt))
        for window in [12,16,24]:
            name=f'I1_{trace}_{year}_w{window}';p=RESULTS/f'{name}.json'
            if p.exists():rows.append(json.loads(p.read_text()));continue
            start=time.perf_counter();zr=[zone_bound(z,arr,refgrid,window,name) for z in range(4)]
            lb=sum(q['conservative_usd'] for q in zr)+float(shared.sum())
            row=dict(experiment='I1',year=year,trace=trace,window_h=window,reference_cost_usd=refcost,lower_bound_usd=lb,maximum_saving_upper_usd=refcost-lb,raw_dual_usd=sum(q['dual_usd'] for q in zr)+float(shared.sum()),seconds=time.perf_counter()-start)
            p.write_text(json.dumps(row,indent=2));rows.append(row)
            zones.extend(dict(year=year,trace=trace,window_h=window,**q) for q in zr)
            print(name,'lower_bound',round(lb,2),'saving_upper',round(refcost-lb,2),flush=True)
pd.DataFrame(rows).to_csv(RESULTS/'cost_bounds.csv',index=False)
pd.DataFrame(zones).to_csv(RESULTS/'cost_bound_zone_diagnostics.csv',index=False)
