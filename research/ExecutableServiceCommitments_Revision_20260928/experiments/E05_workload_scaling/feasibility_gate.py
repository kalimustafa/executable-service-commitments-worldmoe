"""Exact earliest-service deadline check for scaled Eagle arrival profiles."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]
sys.path.insert(0,str(WORKSPACE/'ServiceCommitment_1to5_20260927'))
from src import context as ctx
arr,_,_=ctx.configure(2025)
c=ctx.c
nominal_util=float(np.sum(arr*c.k)/(len(arr)*np.sum(c.k)))
rows=[]
for utilization in (.4,.6,.8):
    scaled=arr*(utilization/nominal_util)
    earliest=ctx.engine.reference(scaled,1.)
    late=ctx.fifo_lateness(earliest,scaled,24)
    closing=float(c.DT*np.sum(earliest[c.N:]*c.k))
    rows.append(dict(target_utilization=utilization,nominal_utilization=nominal_util,
        scale_factor=utilization/nominal_util,earliest_late_it_mwh=late,
        earliest_closing_it_mwh=closing,physically_feasible_24h=late<1e-8,
        implication='No flexible/rigid split can make the unchanged total arrivals feasible at the same capacity and 24 h deadline'))
frame=pd.DataFrame(rows)
frame.to_csv(HERE/'earliest_service_feasibility.csv',index=False)
frame.to_csv(HERE.parents[1]/'source_data/extfig2_workload_feasibility.csv',index=False)
print(frame[['target_utilization','earliest_late_it_mwh','earliest_closing_it_mwh']].to_string(index=False))
