"""Assign an executed fluid book to fixed 4/12/24 h released cohorts.

The class split is fixed before observing prices or policy outcomes. No work
is dropped. EDF and FIFO differ only in attribution of executed service.
"""
from __future__ import annotations
import heapq
import numpy as np

WINDOWS = (4, 12, 24)
SHARES = (.25, .50, .25)
DT = .25

def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    return float(v[np.searchsorted(np.cumsum(w), q * w.sum(), side='left')])

def audit(book, arrivals, site_it_mw, priority='edf'):
    assert priority in ('edf', 'fifo')
    n, sites = arrivals.shape
    queues = [[] for _ in range(sites)]
    released = np.zeros(3)
    timely = np.zeros(3)
    late_work = np.zeros(3)
    closing = np.zeros(3)
    integrated_late = np.zeros(3)
    delays, weights = [], []
    counter = 0
    for t in range(len(book)):
        for z in range(sites):
            if t < n and arrivals[t,z] > 1e-12:
                for cls,(window,share) in enumerate(zip(WINDOWS,SHARES)):
                    amount = float(arrivals[t,z]*share)
                    due = t + 4*window - 1
                    key = due if priority=='edf' else t
                    heapq.heappush(queues[z],(key,counter,t,due,cls,amount))
                    counter += 1
                    released[cls] += amount*site_it_mw[z]*DT
            remaining = float(book[t,z])
            while remaining > 1e-10:
                if not queues[z]:
                    if remaining < 1e-7:break
                    raise RuntimeError(f'Service without released work at step {t}, site {z}')
                key,i,release,due,cls,amount = heapq.heappop(queues[z])
                take = min(remaining,amount)
                remaining -= take
                amount -= take
                energy = take*site_it_mw[z]*DT
                if t <= due:timely[cls] += energy
                else:
                    late_work[cls] += energy
                    integrated_late[cls] += energy*(t-due)
                if t >= n:closing[cls] += energy
                delays.append((t-release)*DT)
                weights.append(energy)
                if amount > 1e-10:heapq.heappush(queues[z],(key,i,release,due,cls,amount))
    residual=sum(item[-1] for queue in queues for item in queue)
    if residual > 1e-5:raise RuntimeError(f'Uncompleted work units: {residual}')
    p={f'p{int(q*100)}_delay_h':weighted_quantile(delays,weights,q) for q in (.5,.9,.95,.99)}
    return dict(priority=priority,total_it_mwh=float(released.sum()),
        late_work_it_mwh=float(late_work.sum()),late_backlog_mwh_quarters=float(integrated_late.sum()),
        closing_it_mwh=float(closing.sum()),
        **{f'{w}h_completion_rate':float(timely[i]/released[i]) if released[i] else 1. for i,w in enumerate(WINDOWS)},
        **{f'{w}h_late_work_it_mwh':float(late_work[i]) for i,w in enumerate(WINDOWS)},
        **p)
