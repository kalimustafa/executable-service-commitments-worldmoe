"""Recompute matched earliest-service references on residual capacity once per cell."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
ENGINE=WORKSPACE/"ExecutableServiceCommitments_Revision_20260928"/"engine"
sys.dont_write_bytecode=True
sys.path.insert(0,str(ENGINE))
from src.runtime import EXTENSION as FROZEN_EXTENSION
sys.path.insert(0,str(FROZEN_EXTENSION));sys.path.insert(0,str(ROOT/"scripts"))
import src,vendor
src.__path__.append(str(FROZEN_EXTENSION/"src"));vendor.__path__.append(str(FROZEN_EXTENSION/"vendor"))
from src import context as ctx
from g3_executor import execute_rigid_flexible_earliest,deadline_miss_mwh,closing_work_mwh

REFERENCE_CACHE={}

def output_files():
    for folder in [ROOT/"outputs/smoke",ROOT/"outputs/primary_seed",ROOT/"outputs/three_seed"]:
        if folder.exists():
            yield from folder.rglob("*.json")

def case_reference(result:dict)->dict:
    name=str(result.get("name",""));smoke=re.search(r"_smoke(\d+)d_z(\d+)",name)
    smoke_days=int(smoke.group(1)) if smoke else 0
    smoke_site=int(smoke.group(2)) if smoke else 0
    key=(result["case_id"],float(result["total_utilization_target"]),float(result["flexible_share"]),smoke_days,smoke_site)
    if key in REFERENCE_CACHE:return REFERENCE_CACHE[key]

    eagle,_,shared=ctx.configure(2025);c=ctx.c
    site_mask=np.ones(4,dtype=bool)
    if smoke_days:
        n=min(smoke_days*96,c.N);site_mask[:]=False;site_mask[smoke_site]=True
        eagle=eagle[:n].copy();shared=shared[:n].copy();c.times=c.times[:n];c.N=n
        for field in ["da","rt","pv","pue"]:
            values=getattr(c,field)[:n].copy();values[:,~site_mask]=0.;setattr(c,field,values)
        eagle[:,~site_mask]=0.;c.k=np.asarray(c.k).copy();c.k[~site_mask]=0.
        c.idle=np.asarray(c.idle).copy();c.idle[~site_mask]=0.
        c.day_index,c.days=pd.factorize(c.times.tz_convert("America/New_York").date,sort=True)
    else:n=c.N
    capacity=np.minimum(1.,(np.asarray(c.c["grid_import_limit"])/np.asarray(c.c["pue_max"])-c.idle)/np.maximum(c.k,1e-12))
    capacity=np.maximum(np.minimum(capacity,1.),0.);capacity[~site_mask]=0.
    dt=float(c.DT);u=float(result["total_utilization_target"]);share=float(result["flexible_share"])
    capacity_energy=dt*n*float(np.sum(capacity*c.k))
    eagle_energy=dt*float(np.sum(eagle*c.k[None,:]))
    flex_arrivals=eagle*((u*share*capacity_energy)/eagle_energy)
    rigid_util=u*(1-share);rigid=rigid_util*capacity
    rigid_arrivals=np.broadcast_to(rigid,(n,4)).copy()
    residual=np.maximum(capacity-rigid,0.)
    total_arrivals=rigid_arrivals+flex_arrivals

    # The rigid fraction is known and executed immediately. The unchanged EDF
    # helper schedules flexible Eagle arrivals on residual capacity, including
    # the terminal continuation, and the same rigid book is then added back.
    flexible_ref,joined,_=execute_rigid_flexible_earliest(np.zeros_like(rigid_arrivals),flex_arrivals,residual,96)
    if not np.allclose(joined,flex_arrivals,rtol=1e-10,atol=1e-9):raise RuntimeError("Reference flexible arrivals changed.")
    reference=flexible_ref.copy();reference[:n]+=rigid[None,:]
    late=deadline_miss_mwh(reference,total_arrivals,c.k,dt,96)
    closing=closing_work_mwh(reference,n,c.k,dt)
    flex_served=dt*float(np.sum(flexible_ref[:n]*c.k[None,:]))
    reference_cash,_=c.cash(reference)
    liability=dt*float(np.sum(reference[n:]*c.k[None,:]*np.max(c.c["pue_max"])*c.train_rt[None,:]))
    ref_cost=float(reference_cash.sum()+shared.sum()+liability)
    record={"reference":reference,"n":n,"late":late,"closing":closing,
            "flex_served":flex_served,"reference_cost":ref_cost,
            "structurally_feasible":bool(late<=1e-8),"residual":residual,"rigid":rigid}
    REFERENCE_CACHE[key]=record
    return record

def refresh_one(path:Path)->dict:
    result=json.loads(path.read_text(encoding="utf-8"))
    ref=case_reference(result)
    result["reference_late_it_mwh"]=ref["late"]
    result["reference_closing_it_mwh"]=ref["closing"]
    result["reference_flexible_served_it_mwh"]=ref["flex_served"]
    result["structurally_feasible"]=ref["structurally_feasible"]
    result["reference_matched_residual_capacity_through_terminal_continuation"]=True
    result["reference_flexible_capacity_fraction"]=float(np.sum(ref["residual"]*ctx.c.k)/max(np.sum(ref["residual"]*ctx.c.k)+np.sum(ref["rigid"]*ctx.c.k),1e-12))
    delta=0.0
    if not str(result.get("name","")).find("smoke")>=0:
        old=float(result["reference_cost_usd"]);delta=ref["reference_cost"]-old
        if abs(delta)>1e-3:
            raise RuntimeError(f"Matched-reference cost changed by {delta:.6f} USD for {path.name}; daily settlement needs review.")
        result["reference_cost_usd"]=ref["reference_cost"]
        if isinstance(result.get("saving_usd"),(float,int)):
            result["policy_cost_usd"]=ref["reference_cost"]-float(result["saving_usd"])
            result["saving_pct_total_cost"]=100.*float(result["saving_usd"])/max(ref["reference_cost"],1e-12)
    result["matched_reference_cost_delta_usd"]=delta
    if not ref["structurally_feasible"] and str(result.get("status"))=="POLICY_REPLAY_COMPLETE":
        result["policy_feasible"]=False
        result["status"]="POLICY_REPLAY_DIAGNOSTIC_STRUCTURAL_REFERENCE_INFEASIBLE"
        result["delivered_saving_usd"]="NA"
    book_path=path.parent/"trajectories"/f"{result.get('name')}_complete_book.npz"
    if book_path.exists():
        with np.load(book_path,allow_pickle=False) as archive:policy=archive["compute"].copy()
        tmp=book_path.with_name(book_path.stem+"_matched.tmp.npz")
        np.savez_compressed(tmp,compute=policy,reference=ref["reference"])
        tmp.replace(book_path)
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return {"case_id":result.get("case_id"),"seed":result.get("seed"),
        "reference_late_it_mwh":ref["late"],"reference_closing_it_mwh":ref["closing"],
        "structurally_feasible":ref["structurally_feasible"],
        "reference_flexible_served_it_mwh":ref["flex_served"],"reference_cost_delta_usd":delta,
        "reference_matched_to_residual_capacity":True,"result_file":str(path.relative_to(ROOT))}

def refresh_all():
    audits=[refresh_one(p) for p in output_files()]
    out=ROOT/"outputs/diagnostics/matched_reference_screen.csv"
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audits).to_csv(out,index=False,encoding="utf-8-sig")
    return audits,out

def main():
    audits,out=refresh_all()
    print(f"Matched residual-capacity references refreshed: {len(audits)} saved runs; {out}")

if __name__=="__main__":main()
