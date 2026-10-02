"""Nonanticipative current action with forecast-scenario continuation values."""
# Imported integral implementation stamp: 20260928-v2. Numba's disk cache does
# not track imported helper edits; update this stamp when continuous_cost changes.
import numpy as np
from numba import njit
from src.models.continuous_cost import continuous_curve

@njit(cache=True)
def _curve(pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref,capacity):
    s=len(pv);knots=np.empty(s+3);knots[0]=0.;knots[1]=capacity;knots[2]=min(capacity,max(0.,reference))
    for j in range(s):knots[j+3]=min(capacity,max(0.,(pv[j]/pue-idle)/power))
    knots.sort();weight=np.zeros(s+2);slope=np.zeros(s+2);count=0
    for j in range(len(knots)-1):
        w=knots[j+1]-knots[j]
        if w<1e-12:continue
        mid=.5*(knots[j+1]+knots[j]);load=pue*(idle+power*mid);value=0.
        for k in range(s):
            grid=load>pv[k]
            derivative=prices[k] if grid else -1.
            if grid:derivative+=1. if mid>=reference else -1.
            value+=probabilities[k]*derivative
        value=(value*pue*power+opportunity*pue_ref*power)*.25
        weight[count]=w;slope[count]=value;count+=1
        while count>1 and slope[count-2]>slope[count-1]+1e-12:
            total=weight[count-2]+weight[count-1]
            slope[count-2]=(slope[count-2]*weight[count-2]+slope[count-1]*weight[count-1])/total
            weight[count-2]=total;count-=1
    return slope[:count],weight[:count]

@njit(cache=True)
def _future_selected(queue,pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref,capacity,continuous=False,mass_aware=False):
    scenarios,h=pv.shape;maximum=h*(scenarios+9 if continuous else scenarios+2)
    # No slot can receive more than the total work still being planned. This
    # redundant bound preserves the feasible set while concentrating the
    # continuous-cost discretization on service amounts that can actually occur.
    curve_capacity=min(capacity,queue.sum()) if mass_aware else capacity
    costs=np.zeros(maximum);caps=np.zeros(maximum);slots=np.zeros(maximum,np.int64);n=0
    for t in range(1,h):
        if continuous:c,w=continuous_curve(pv[:,t],prices[:,t],probabilities,pue[t],reference[t],idle,power,opportunity[t],pue_ref,curve_capacity)
        else:c,w=_curve(pv[:,t],prices[:,t],probabilities,pue[t],reference[t],idle,power,opportunity[t],pue_ref,curve_capacity)
        for k in range(len(c)):costs[n]=c[k];caps[n]=w[k];slots[n]=t;n+=1
    heap=np.empty(n,np.int64);heap_n=0;cursor=0;selected=np.zeros(n)
    for due in range(h):
        while cursor<n and slots[cursor]<=due:
            k=heap_n;heap_n+=1;item=cursor;cursor+=1
            while k>0:
                parent=(k-1)//2;old=heap[parent]
                if costs[old]<costs[item] or (costs[old]==costs[item] and old<item):break
                heap[k]=old;k=parent
            heap[k]=item
        remaining=queue[due]
        while remaining>1e-10 and heap_n:
            item=heap[0];take=min(remaining,caps[item]);remaining-=take;caps[item]-=take;selected[item]+=take
            if caps[item]<=1e-10:
                heap_n-=1
                if not heap_n:break
                replacement=heap[heap_n];k=0
                while 2*k+1<heap_n:
                    child=2*k+1
                    if child+1<heap_n:
                        left=heap[child];right=heap[child+1]
                        if costs[right]<costs[left] or (costs[right]==costs[left] and right<left):child+=1
                    nxt=heap[child]
                    if costs[replacement]<costs[nxt] or (costs[replacement]==costs[nxt] and replacement<nxt):break
                    heap[k]=nxt;k=child
                heap[k]=replacement
        if remaining>1e-7:return np.zeros(1),np.array([1e12])
    order=np.argsort(costs[:n])[::-1]
    keep=0
    for j in order:
        if selected[j]>1e-12:keep+=1
    selected_costs=np.empty(keep);selected_amounts=np.empty(keep);k=0
    for j in order:
        if selected[j]>1e-12:
            selected_costs[k]=costs[j];selected_amounts[k]=selected[j];k+=1
    return selected_costs,selected_amounts

@njit(cache=True)
def _removed_value(costs,amounts,work):
    value=0.
    for k in range(len(costs)):
        take=min(work,amounts[k]);work-=take;value+=take*costs[k]
        if work<1e-10:break
    return value

@njit(cache=True)
def _current_cost(x,pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref):
    result=0.;base=pue*idle;load=pue*(idle+power*x);ref=pue*(idle+power*reference)
    for s in range(len(pv)):
        used0=min(pv[s],base);used=min(pv[s],load);grid0=base-used0;grid=load-used;refgrid=max(0.,ref-pv[s])
        result+=probabilities[s]*(prices[s]*(grid-grid0)+(used0-used)+abs(grid-refgrid)-abs(grid0-refgrid))
    return result*.25+opportunity*pue_ref*power*x*.25

@njit(cache=True)
def action_value_curves(queue,pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref,capacity,safety=.9,need_recourse=True,exact_knots=False,continuous=False,mass_aware=False):
    total=queue.sum();upper=min(capacity,total)
    if upper<1e-10:return np.zeros(1),np.zeros(1),np.zeros(1)
    prefix=0.;required=0.
    for due in range(len(queue)):
        prefix+=queue[due];required=max(required,prefix-safety*capacity*due)
    lower=min(upper,max(0.,required))
    if upper-lower<1e-9:return np.array([upper]),np.zeros(1),np.zeros(1)
    residual=queue.copy();left=lower
    for d in range(len(queue)):
        take=min(left,residual[d]);residual[d]-=take;left-=take
    c_ev,w_ev=_future_selected(residual,pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref,capacity,continuous,mass_aware)
    s=len(pv);scenario_costs=np.zeros((s,len(queue)*3));scenario_amounts=np.zeros_like(scenario_costs);lengths=np.zeros(s,np.int64)
    if need_recourse:
        for k in range(s):
            c,w=_future_selected(residual,pv[k:k+1],prices[k:k+1],np.ones(1),pue,reference,idle,power,opportunity,pue_ref,capacity)
            lengths[k]=len(c);scenario_costs[k,:len(c)]=c;scenario_amounts[k,:len(c)]=w
    maximum=3+s+len(w_ev)+int(lengths.sum()) if exact_knots else 55+s
    candidates=np.empty(maximum);n=0;candidates[n]=lower;n+=1;candidates[n]=upper;n+=1
    if exact_knots:
        # The objective is piecewise linear. Its complete breakpoint union is
        # sufficient for a global minimum, including negative-price pieces.
        value=lower
        for amount in w_ev:
            value+=amount
            if value>=upper-1e-12:break
            if value>lower+1e-12:candidates[n]=value;n+=1
        if need_recourse:
            for k in range(s):
                value=lower
                for j in range(lengths[k]):
                    value+=scenario_amounts[k,j]
                    if value>=upper-1e-12:break
                    if value>lower+1e-12:candidates[n]=value;n+=1
    else:
        value=lower+.025
        while value<upper:
            candidates[n]=value;n+=1;value+=.025
    for k in range(s):
        value=(pv[k,0]/pue[0]-idle)/power
        if lower<value<upper:candidates[n]=value;n+=1
    if lower<reference[0]<upper:candidates[n]=reference[0];n+=1
    base=np.zeros(n);adjustment=np.zeros(n)
    for j in range(n):
        x=candidates[j];removed_ev=_removed_value(c_ev,w_ev,x-lower);removed_recourse=0.
        if need_recourse:
            for k in range(s):removed_recourse+=probabilities[k]*_removed_value(scenario_costs[k,:lengths[k]],scenario_amounts[k,:lengths[k]],x-lower)
        base[j]=_current_cost(x,pv[:,0],prices[:,0],probabilities,pue[0],reference[0],idle,power,opportunity[0],pue_ref)-removed_ev
        adjustment[j]=removed_ev-removed_recourse
    return candidates[:n],base,adjustment

@njit(cache=True)
def select_action(queue,pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref,capacity,eta=1.,safety=.9,exact_knots=False,continuous=False,mass_aware=False):
    candidates,base,adjustment=action_value_curves(queue,pv,prices,probabilities,pue,reference,idle,power,opportunity,pue_ref,capacity,safety,eta!=0.,exact_knots,continuous,mass_aware)
    best=1e100;answer=candidates[0]
    for j in range(len(candidates)):
        x=candidates[j];cost=base[j]+eta*adjustment[j]
        if cost<best-1e-9 or (abs(cost-best)<1e-9 and x>answer):best=cost;answer=x
    return answer

@njit(cache=True)
def realized_action_costs(candidates,queue,rt,pv,pue,reference,idle,power,pue_ref,capacity,safety=.9):
    prefix=0.;required=0.
    for due in range(len(queue)):
        prefix+=queue[due];required=max(required,prefix-safety*capacity*due)
    lower=min(min(capacity,queue.sum()),max(0.,required));remaining=queue.copy();left=lower
    for d in range(len(queue)):
        take=min(left,remaining[d]);remaining[d]-=take;left-=take
    if len(candidates)==1:return np.zeros(1)
    zero=np.zeros(len(queue));prob=np.ones(1)
    costs,amounts=_future_selected(remaining,pv[None,:],rt[None,:],prob,pue,reference,idle,power,zero,pue_ref,capacity)
    target=np.zeros(len(candidates))
    for j,x in enumerate(candidates):
        target[j]=_current_cost(x,pv[:1],rt[:1],prob,pue[0],reference[0],idle,power,0.,pue_ref)-_removed_value(costs,amounts,x-lower)
    return target


@njit(cache=True)
def forecast_policy_action_costs(candidates,queue,pv_scenarios,prices,probabilities,pue_forecast,reference_forecast,opportunity,rt,pv,pue,reference_actual,idle,power,pue_ref,capacity,safety=.9):
    """Evaluate forecast-selected plans using actual labels, without hindsight planning."""
    h=len(queue);s=len(probabilities);prefix=0.;required=0.
    for due in range(h):
        prefix+=queue[due];required=max(required,prefix-safety*capacity*due)
    lower=min(min(capacity,queue.sum()),max(0.,required));residual=queue.copy();left=lower
    for d in range(h):
        take=min(left,residual[d]);residual[d]-=take;left-=take
    maximum=h*(s+2);costs=np.zeros(maximum);widths=np.zeros(maximum);slots=np.zeros(maximum,np.int64);count=0
    for t in range(1,h):
        c,w=_curve(pv_scenarios[:,t],prices[:,t],probabilities,pue_forecast[t],reference_forecast[t],idle,power,opportunity[t],pue_ref,capacity)
        for j in range(len(c)):costs[count]=c[j];widths[count]=w[j];slots[count]=t;count+=1
    order=np.argsort(costs[:count]);chosen=np.zeros(count)
    for due in range(h):
        remaining=residual[due]
        for j in order:
            if slots[j]>due:continue
            take=min(remaining,widths[j]-chosen[j]);chosen[j]+=take;remaining-=take
            if remaining<1e-10:break
        if remaining>1e-7:raise ValueError('Infeasible forecast-plan training state')
    y=np.zeros(len(candidates));one=np.ones(1)
    for a,x in enumerate(candidates):
        current=chosen.copy();left=x-lower
        for j in order[::-1]:
            take=min(left,current[j]);current[j]-=take;left-=take
            if left<1e-10:break
        plan=np.zeros(h);plan[0]=x
        for j in range(count):plan[slots[j]]+=current[j]
        if abs(plan.sum()-queue.sum())>1e-6:raise ValueError('Training label dropped workload')
        for t in range(h):
            y[a]+=_current_cost(plan[t],pv[t:t+1],rt[t:t+1],one,pue[t],reference_actual[t],idle,power,0.,pue_ref)
    return y
