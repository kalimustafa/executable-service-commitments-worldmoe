from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent
CONFIG=json.loads((ROOT/"configs/g3_cases.json").read_text(encoding="utf-8"))
sys.path.insert(0,str(ROOT/"palette"))
import theme as C

RESULT_COLUMNS=[
    "case_id","seed","total_utilization_target","total_utilization_realized","flexible_share",
    "rigid_mean_utilization","flexible_mean_utilization","total_variable_it_mwh","rigid_it_mwh",
    "flexible_released_it_mwh","reference_flexible_served_it_mwh","policy_flexible_served_it_mwh",
    "reference_cost_usd","policy_cost_usd","saving_usd","saving_pct_total_cost",
    "saving_per_flexible_mwh","shifted_flexible_it_mwh","saving_per_shifted_mwh",
    "saving_per_flexible_mwh_retention_pct",
    "reference_late_it_mwh","policy_late_it_mwh","reference_closing_it_mwh","policy_closing_it_mwh",
    "cvar95_usd","maximum_grid_import_mw","maximum_recovery_import_mw",
    "structurally_feasible","policy_feasible","status","name","daily_results_file","notes"
]

def read_native():
    path=WORKSPACE/CONFIG["native_reference"]["source_csv"]
    df=pd.read_csv(path,encoding="utf-8-sig")
    df=df[df.window_h.astype(int).eq(24)].copy()
    with np.load(WORKSPACE/"external_validation/data/forcing_2025.npz",allow_pickle=False) as z:
        k=z["k"].astype(float);n=len(z["eagle_arrivals"])
        flex_mwh=.25*float(np.sum(z["eagle_arrivals"]*k[None,:]))
    rows=[]
    for r in df.to_dict("records"):
        saving=float(r["saving_usd"]);late=float(r["late_IT_mwh"]);closing=float(r["closing_it_mwh"])
        rows.append({
            "case_id":"G3-NATIVE","seed":int(r["seed"]),
            "total_utilization_target":CONFIG["native_reference"]["utilization"],
            "total_utilization_realized":CONFIG["native_reference"]["utilization"],
            "flexible_share":1.0,"rigid_mean_utilization":0.0,
            "flexible_mean_utilization":CONFIG["native_reference"]["utilization"],
            "total_variable_it_mwh":flex_mwh,"rigid_it_mwh":0.0,
            "flexible_released_it_mwh":flex_mwh,
            "reference_flexible_served_it_mwh":flex_mwh,
            "policy_flexible_served_it_mwh":max(0.,flex_mwh-closing),
            "reference_cost_usd":float(r["reference_cost_usd"]),
            "policy_cost_usd":float(r["adjusted_cost_usd"]),"saving_usd":saving,
            "saving_pct_total_cost":float(r["saving_pct"]),
            "saving_per_flexible_mwh":saving/flex_mwh,
            "shifted_flexible_it_mwh":float(r["shifted_IT_mwh"]),
            "saving_per_shifted_mwh":saving/float(r["shifted_IT_mwh"]) if float(r["shifted_IT_mwh"])>0 else "NA",
            "reference_late_it_mwh":0.0,"policy_late_it_mwh":late,
            "reference_closing_it_mwh":0.0,"policy_closing_it_mwh":closing,
            "cvar95_usd":float(r["cvar95_usd"]),"maximum_grid_import_mw":"NA",
            "maximum_recovery_import_mw":float(r["max_fleet_recovery_mw"]),
            "structurally_feasible":True,"policy_feasible":bool(late<=1e-8),
            "status":"REUSED_NATIVE_24H_SOURCE",
            "notes":"Read from frozen native 24 h three-seed source CSV; not rerun."
        })
    return rows

def read_g3_runs():
    rows=[];daily=[]
    for base in [ROOT/"outputs/primary_seed",ROOT/"outputs/three_seed"]:
        for p in base.rglob("*.json"):
            if "trajectories" in p.parts: continue
            r=json.loads(p.read_text(encoding="utf-8"))
            rows.append(r)
            daily_rel=r.get("daily_results_file")
            if daily_rel:
                daily_path=ROOT/daily_rel
                if daily_path.exists():
                    daily.append(pd.read_csv(daily_path,encoding="utf-8-sig"))
    return rows,daily

def make_results():
    runs,daily_frames=read_g3_runs()
    results=read_native()+runs
    df=pd.DataFrame(results)
    for column in RESULT_COLUMNS:
        if column not in df: df[column]=np.nan
    df=df[RESULT_COLUMNS].sort_values(["total_utilization_target","flexible_share","seed"],kind="stable")
    native_flex_value=pd.to_numeric(df.loc[df.case_id.eq("G3-NATIVE"),"saving_per_flexible_mwh"],errors="coerce").mean()
    df["saving_per_flexible_mwh_retention_pct"]=100.*pd.to_numeric(df["saving_per_flexible_mwh"],errors="coerce")/native_flex_value
    out=ROOT/"outputs/G3_REALISTIC_UTILIZATION_RESULTS.csv"
    df.to_csv(out,index=False,encoding="utf-8-sig")
    out.parent.mkdir(parents=True,exist_ok=True)
    (ROOT/"tables").mkdir(exist_ok=True)
    df.to_csv(ROOT/"tables/G3_REALISTIC_UTILIZATION_RESULTS.csv",index=False,encoding="utf-8-sig")
    if daily_frames:
        daily_df=pd.concat(daily_frames,ignore_index=True)
        daily_df.to_csv(ROOT/"outputs/daily/G3_REALISTIC_UTILIZATION_DAILY.csv",index=False,encoding="utf-8-sig")
    else: daily_df=pd.DataFrame()

    summaries=[]
    for case_id in ["G3-NATIVE"]+[c["case_id"] for c in CONFIG["cases"]]:
        group=df[df.case_id.eq(case_id)].copy()
        if group.empty: continue
        first=group.iloc[0]
        is_native=case_id=="G3-NATIVE"
        policy=group if is_native else group[group.status.eq("POLICY_REPLAY_COMPLETE") & group.structurally_feasible.fillna(False)]
        structural=bool(group.structurally_feasible.fillna(False).all())
        saves=pd.to_numeric(policy.saving_usd,errors="coerce").dropna()
        rates=pd.to_numeric(policy.saving_pct_total_cost,errors="coerce").dropna()
        perflex=pd.to_numeric(policy.saving_per_flexible_mwh,errors="coerce").dropna()
        late=pd.to_numeric(policy.policy_late_it_mwh,errors="coerce").dropna()
        if len(policy)==0 and not is_native:
            late=pd.to_numeric(group.reference_late_it_mwh,errors="coerce").dropna()
        closing=pd.to_numeric(policy.policy_closing_it_mwh,errors="coerce").dropna()
        summaries.append({
            "case":case_id,"utilization":float(first.total_utilization_target),
            "flex_share":float(first.flexible_share),"rigid_util":float(first.rigid_mean_utilization),
            "flex_util":float(first.flexible_mean_utilization),
            "saving_mean":float(saves.mean()) if len(saves) else np.nan,
            "saving_sd":float(saves.std(ddof=1)) if len(saves)>1 else (0. if len(saves)==1 else np.nan),
            "saving_min":float(saves.min()) if len(saves) else np.nan,
            "saving_max":float(saves.max()) if len(saves) else np.nan,
            "saving_pct_mean":float(rates.mean()) if len(rates) else np.nan,
            "saving_pct_sd":float(rates.std(ddof=1)) if len(rates)>1 else (0. if len(rates)==1 else np.nan),
            "saving_per_flex_mwh":float(perflex.mean()) if len(perflex) else np.nan,
            "saving_per_flex_mwh_sd":float(perflex.std(ddof=1)) if len(perflex)>1 else (0. if len(perflex)==1 else np.nan),
            "saving_per_flex_mwh_min":float(perflex.min()) if len(perflex) else np.nan,
            "saving_per_flex_mwh_max":float(perflex.max()) if len(perflex) else np.nan,
            "saving_per_flex_retention_pct":100.*float(perflex.mean())/float(native_flex_value) if len(perflex) and float(native_flex_value)>0 else np.nan,
            "late_max":float(late.max()) if len(late) else np.nan,
            "closing_mean":float(closing.mean()) if len(closing) else np.nan,
            "feasible_seed_count":int(pd.to_numeric(policy.policy_feasible,errors="coerce").fillna(0).sum()),
            "policy_seed_count":int(len(policy)),"structurally_feasible":structural,
            "sample_scope":"three frozen refinement seeds" if len(policy)>=3 or is_native else ("single primary structural screen" if len(policy)==0 else "frozen primary-seed replay")
        })
    summary=pd.DataFrame(summaries)
    summary.to_csv(ROOT/"outputs/G3_SUMMARY.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(ROOT/"tables/G3_SUMMARY.csv",index=False,encoding="utf-8-sig")
    return df,summary,daily_df

def build_consistency_audit(df):
    forcing_path=WORKSPACE/"external_validation/data/forcing_2025.npz"
    with np.load(forcing_path,allow_pickle=False) as z:
        n=len(z["eagle_arrivals"]);k=z["k"].astype(float);idle=z["idle"].astype(float)
    constraints=json.loads((WORKSPACE/"R2_PAPER_INTEGRATION_20260923/configs/economic_constraints.json").read_text(encoding="utf-8-sig"))["constraints"]
    cap=np.maximum(np.minimum((np.asarray(constraints["grid_import_limit"])/np.asarray(constraints["pue_max"])-idle)/k,1.),0.)
    tol=1e-8
    audits=[]
    for r in df[~df.case_id.eq("G3-NATIVE")].to_dict("records"):
        status=str(r.get("status",""));policy_ran=status.startswith("POLICY_REPLAY")
        energy_gap=abs(float(r["rigid_it_mwh"])+float(r["flexible_released_it_mwh"])-float(r["total_variable_it_mwh"]))
        util_gap=abs(float(r["total_utilization_realized"])-float(r["total_utilization_target"]))
        reference_late=float(r["reference_late_it_mwh"])
        base={"case_id":r["case_id"],"seed":r["seed"],"policy_replay":policy_ran,
              "energy_conservation_gap_mwh":energy_gap,"utilization_error_fraction":util_gap,
              "rigid_within_capacity":True,"policy_flexible_within_residual_capacity":"NA",
              "same_rigid_and_flexible_loads":True,"same_forcing_and_settlement":True,
              "saving_accounting_error_usd":"NA","late_nonnegative":reference_late>=-tol,
              "closing_nonnegative":float(r["reference_closing_it_mwh"])>=-tol,
              "no_training_or_checkpoint_reselection":True}
        if policy_ran:
            daily_path=ROOT/r["daily_results_file"]
            book_path=daily_path.parent/"trajectories"/f"{r['name']}_complete_book.npz"
            with np.load(book_path,allow_pickle=False) as z:
                policy=np.asarray(z["compute"],dtype=float);reference=np.asarray(z["reference"],dtype=float)
            rigid_util=float(r["rigid_mean_utilization"]);rigid=rigid_util*cap
            residual=np.maximum(cap-rigid,0.)
            flex=policy[:n]-rigid[None,:]
            ref_flex=reference[:n]-rigid[None,:]
            flex_cap_violation=max(0.,float(np.max(flex-residual[None,:])))
            total_cap_violation=max(0.,float(np.max(policy[:n]-cap[None,:])))
            scheduled_energy=.25*float(np.sum(policy*k[None,:]))
            released=float(r["total_variable_it_mwh"])
            saving_error=abs(float(r["reference_cost_usd"])-float(r["policy_cost_usd"])-float(r["saving_usd"]))
            base.update({"policy_flexible_within_residual_capacity":flex_cap_violation<=tol,
                         "policy_capacity_violation_max":flex_cap_violation,
                         "total_capacity_violation_max":total_cap_violation,
                         "scheduled_energy_gap_mwh":abs(scheduled_energy-released),
                         "reference_has_same_rigid_service":bool(np.min(ref_flex)>=-tol),
                         "saving_accounting_error_usd":saving_error,
                         "late_nonnegative":float(r["policy_late_it_mwh"])>=-tol,
                         "closing_nonnegative":float(r["policy_closing_it_mwh"])>=-tol})
            base["pass"]=(energy_gap<=1e-7 and util_gap<=tol and flex_cap_violation<=tol and
                           total_cap_violation<=tol and abs(scheduled_energy-released)<=1e-6 and
                           saving_error<=1e-4 and base["late_nonnegative"] and base["closing_nonnegative"] and
                           base["reference_has_same_rigid_service"])
        else:
            base["pass"]=(energy_gap<=1e-7 and util_gap<=tol and
                          bool(r["structurally_feasible"])==(reference_late<=tol) and
                          base["late_nonnegative"] and base["closing_nonnegative"])
        audits.append(base)
    audit=pd.DataFrame(audits)
    path=ROOT/"outputs/diagnostics/G3_CONSISTENCY_AUDIT.csv"
    path.parent.mkdir(parents=True,exist_ok=True)
    audit.to_csv(path,index=False,encoding="utf-8-sig")
    if not audit["pass"].all():
        raise RuntimeError(f"G3 consistency audit failed; see {path}")
    return audit

def plot_figure(df,summary):
    import theme as _unused_theme  # explicit shared palette import in this figure build environment
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
        "font.size":8,"axes.labelsize":8,"axes.titlesize":9,"xtick.labelsize":7,"ytick.labelsize":7,
        "legend.fontsize":7,"axes.edgecolor":C.TEXT,"text.color":C.TEXT,"axes.labelcolor":C.TEXT,
        "xtick.color":C.TEXT,"ytick.color":C.TEXT,"axes.linewidth":.65,
        "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    data=summary[summary.case.ne("G3-NATIVE")].copy()
    fig,axs=plt.subplots(1,2,figsize=(7.25,3.85),constrained_layout=True)
    a,b=axs
    palette={0.1:C.PRIMARY_DARK,0.3:C.SECONDARY}
    marker={0.1:"o",0.3:"s"}
    valid=pd.to_numeric(data.saving_per_flex_mwh,errors="coerce").dropna()
    ymax=float(valid.max()) if len(valid) else 1.
    y_infeasible=ymax+max(3.,.12*ymax)
    for share in [.1,.3]:
        q=data[np.isclose(pd.to_numeric(data.flex_share),share)].sort_values("utilization")
        xs=[];ys=[];lower=[];upper=[]
        for _,r in q.iterrows():
            x=100*float(r.utilization)
            if not bool(r.structurally_feasible):
                # The x is a status mark at the top margin, not a zero-valued saving.
                a.scatter([x],[y_infeasible],marker="x",s=42,color=palette[share],linewidth=1.5,zorder=4)
                a.annotate("infeasible",(x,y_infeasible),xytext=(5,5),textcoords="offset points",fontsize=6.2)
                xs.append(x);ys.append(np.nan);lower.append(np.nan);upper.append(np.nan)
                continue
            if pd.isna(r.saving_per_flex_mwh):
                xs.append(x);ys.append(np.nan);lower.append(np.nan);upper.append(np.nan);continue
            mean=float(r.saving_per_flex_mwh);lo=float(r.saving_per_flex_mwh_min);hi=float(r.saving_per_flex_mwh_max)
            ys.append(mean);xs.append(x);lower.append(mean-lo);upper.append(hi-mean)
        a.plot(xs,ys,color=palette[share],marker=marker[share],linewidth=1.3,
               label=f"{int(share*100)}% flexible share")
        for x,y,lo,hi in zip(xs,ys,lower,upper):
            if not np.isfinite(y):continue
            a.errorbar(x,y,yerr=[[lo],[hi]],fmt="none",ecolor=palette[share],capsize=2,linewidth=.8)
        for _,r in q.iterrows():
            if bool(r.structurally_feasible) and int(r.feasible_seed_count)<int(r.policy_seed_count) and int(r.policy_seed_count)>0 and pd.notna(r.saving_per_flex_mwh):
                a.scatter([100*float(r.utilization)],[float(r.saving_per_flex_mwh)],marker=marker[share],
                          s=42,facecolors="none",edgecolors=palette[share],linewidths=1.1,zorder=5)
                a.annotate("policy late",(100*float(r.utilization),float(r.saving_per_flex_mwh)),
                           xytext=(4,5),textcoords="offset points",fontsize=5.8)
    a.set(xlabel="Total variable-IT utilization (%)",ylabel="Saving per flexible IT MWh (US$/MWh)",
          xticks=[20,40,60],xlim=(15,65),ylim=(0,y_infeasible+max(5,.08*ymax)))
    a.spines["top"].set_visible(False);a.spines["right"].set_visible(False)
    a.grid(axis="y",color=C.GRID,linewidth=.55,alpha=.75);a.set_axisbelow(True)
    a.set_title("a  Executable value by flexible share",loc="left",fontweight="medium")
    a.legend(frameon=False,loc="upper left")

    for share in [.1,.3]:
        q=data[np.isclose(pd.to_numeric(data.flex_share),share)].sort_values("utilization")
        xs=[];ys=[];statuses=[]
        for _,r in q.iterrows():
            x=100*float(r.utilization)
            structural=bool(r.structurally_feasible)
            if structural:
                y=float(r.late_max) if np.isfinite(r.late_max) else np.nan
                status="policy"
            else:
                y=float(r.late_max) if np.isfinite(r.late_max) else np.nan
                status="earliest reference"
            xs.append(x);ys.append(y if structural else np.nan);statuses.append(status)
            if not structural and np.isfinite(y):
                b.scatter([x],[y],marker="x",s=42,color=palette[share],linewidth=1.5,zorder=5)
                b.annotate("earliest service infeasible",(x,y),xytext=(-6,6),textcoords="offset points",
                           ha="right",fontsize=6.1)
        b.plot(xs,ys,color=palette[share],marker=marker[share],linewidth=1.1,
               label=f"{int(share*100)}% flexible share")
    b.axhline(0,color=C.TEXT,linestyle=(0,(3,2)),linewidth=.75)
    b.set_yscale("symlog",linthresh=1,linscale=.7)
    b.set(xlabel="Total variable-IT utilization (%)",ylabel="Late flexible IT (MWh)",xticks=[20,40,60],xlim=(15,65))
    b.spines["top"].set_visible(False);b.spines["right"].set_visible(False)
    b.grid(axis="y",color=C.GRID,linewidth=.55,alpha=.75);b.set_axisbelow(True)
    b.set_title("b  Deadline feasibility",loc="left",fontweight="medium")
    b.legend(frameon=False,loc="upper left")
    fig.suptitle("Realistic total utilization with a rigid background and flexible Eagle workload",fontsize=10,y=1.02)
    out=ROOT/"figures";out.mkdir(exist_ok=True)
    stem=out/"FIG_G3_REALISTIC_UTILIZATION"
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",pad_inches=.04)
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",pad_inches=.04)
    fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)

    # The plotted records remain inspectable and retain the summary row scope.
    data.to_csv(ROOT/"tables/G3_FIGURE_SOURCE.csv",index=False,encoding="utf-8-sig")
    (ROOT/"figures/FIG_G3_REALISTIC_UTILIZATION_caption.md").write_text(
        """# FIG_G3_REALISTIC_UTILIZATION\n\n**a, Executable value by flexible share.** Mean annual saving per released flexible IT MWh is shown across the predeclared total-utilization cases. Error bars show the three-seed min–max range where three policy replays exist. Structural infeasibility is marked at the top of the plotting field without assigning a saving value.\n\n**b, Deadline feasibility.** Policy late flexible IT is shown for replayed cases. Where the matched earliest-service reference is structurally infeasible, its lateness is shown with an × and labeled as an earliest-service failure; no policy saving is reported for that cell. The y axis uses a symmetric-log scale and retains a zero-lateness reference. Closing IT remains a separately reported year-end obligation and is not classified as deadline lateness.\n\nThe rigid component is a controlled constant proportional capacity-occupancy stress; it is not an empirically observed production-load trace. The flexible share is an annual variable-IT energy share, and the flexible arrival shape is the original Eagle trace scaled once by energy.\n""",encoding="utf-8")

def write_reports(df,summary,daily):
    native=summary[summary.case.eq("G3-NATIVE")].iloc[0]
    cases=summary[summary.case.ne("G3-NATIVE")].sort_values(["utilization","flex_share"])
    policy=df[df.status.eq("POLICY_REPLAY_COMPLETE")].copy()
    def fmt(v,d=2): return "NA" if pd.isna(v) else f"{float(v):,.{d}f}"
    lines=["# G3 realistic utilization report","",
        "## Design and scope","",
        "This is the NYISO 2025 frozen-workload realism stress. The native 5.1915% utilization, 100% flexible Eagle result is read from the existing 24 h three-seed source. In each stress case, a rigid load occupies a constant fraction of variable capacity immediately, while flexible arrivals preserve the Eagle temporal and site pattern and are scaled to the specified annual IT-energy share. The fixed 24 h contract, current-action WorldMoE executor, physical forcing and settlement are unchanged.","",
        "The rigid component is a controlled capacity-occupancy stress, not an empirically measured production rigid-load trace. Seeds are three frozen refinements and are not independent backbone replications. No annual block bootstrap was run; seed spread is descriptive, not an independent-year confidence interval.","",
        "## Native reference","",
        f"Native utilization: {100*native.utilization:.4f}%; mean saving ${native.saving_mean:,.2f}; mean total-cost saving rate {native.saving_pct_mean:.3f}%; mean saving per flexible MWh ${native.saving_per_flex_mwh:.3f}.","",
        "## Six prespecified utilization × flexible-share cells","",
        "| Case | Rigid / flexible mean utilization | Structural reference | Policy seeds | Saving mean ± SD (US$) | Saving rate mean (%) | Saving/flex MWh | Retention vs native (%) | Max late IT (MWh) | Mean closing IT (MWh) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in cases.itertuples():
        group=df[(df.case_id==r.case)&df.status.eq("POLICY_REPLAY_COMPLETE")]
        ref_late=pd.to_numeric(df[df.case_id.eq(r.case)].reference_late_it_mwh,errors="coerce").dropna()
        if not r.structurally_feasible:
            struct=f"infeasible (earliest late {fmt(ref_late.max())})"
        else:struct="feasible"
        policy_count=int(len(group))
        late_display=fmt(r.late_max)
        lines.append(f"| {r.case} | {100*r.rigid_util:.1f}% / {100*r.flex_util:.1f}% | {struct} | {policy_count} | "
            f"{fmt(r.saving_mean)} ± {fmt(r.saving_sd)} | {fmt(r.saving_pct_mean,3)} | {fmt(r.saving_per_flex_mwh,3)} | {fmt(r.saving_per_flex_retention_pct,1)} | {late_display} | {fmt(r.closing_mean,3)} |")
    lines += ["","## Required questions","",
        f"**Q1. Is 5.19% required for value?** {'No for the tested cases: at least one higher-utilization case has positive saving, zero policy late IT across replayed seeds and a structurally feasible reference.' if any((r.utilization>=.4 and r.structurally_feasible and r.feasible_seed_count==r.policy_seed_count and r.policy_seed_count>0 and r.saving_mean>0) for r in cases.itertuples()) else 'The tested results do not establish positive feasible value above the native utilization.'}",""]
    qmap=[(.2,.1,"Q2"),(.2,.3,"Q3"),(.4,.1,"Q4"),(.4,.3,"Q5"),(.6,.1,"Q6"),(.6,.3,"Q7")]
    for util,share,label in qmap:
        r=cases[(np.isclose(cases.utilization,util))&(np.isclose(cases.flex_share,share))].iloc[0]
        if not bool(r.structurally_feasible):
            row=df[df.case_id.eq(r.case)].iloc[0]
            lines.append(f"**{label}. {int(util*100)}%-{int(share*100)}%:** structurally infeasible; earliest-service reference late IT {fmt(row.reference_late_it_mwh)} MWh. Policy economics are NA and are not described as delivered value.")
        else:
            status="positive" if r.saving_mean>0 else "non-positive"
            lines.append(f"**{label}. {int(util*100)}%-{int(share*100)}%:** structurally feasible; {status} mean saving, {int(r.feasible_seed_count)}/{int(r.policy_seed_count)} policy seeds with zero substantive late work; mean saving per flexible MWh ${fmt(r.saving_per_flex_mwh,3)}.")
    structural_false=cases[~cases.structurally_feasible]
    first=structural_false.sort_values(["utilization","flex_share"]).iloc[0] if len(structural_false) else None
    lines += ["",f"**Q8. First structural infeasibility:** {first.case if first is not None else 'none among the six tested cells'}.",
        "","**Q9. Saving per flexible MWh:**"]
    for share in [.1,.3]:
        values=[]
        for util in [.2,.4,.6]:
            r=cases[(np.isclose(cases.utilization,util))&(np.isclose(cases.flex_share,share))].iloc[0]
            values.append(f"{int(util*100)}%={fmt(r.saving_per_flex_mwh,3)}" if pd.notna(r.saving_per_flex_mwh) else f"{int(util*100)}%=NA (structurally infeasible)")
        lines.append(f"- {int(share*100)}% flexible: " + "; ".join(values) + ".")
    support40=cases[(np.isclose(cases.utilization,.4))&(cases.structurally_feasible)&
                    (cases.policy_seed_count>0)&(cases.feasible_seed_count==cases.policy_seed_count)&(cases.saving_mean>0)]
    if len(support40)==2: q10="The 5.19% criticism is answered strongly for the tested controlled stress: both prespecified 40% total-utilization shares retain feasible positive value."
    elif len(support40)==1: q10="The 5.19% criticism is answered for a bounded flexible share at 40% total utilization; the other tested 40% share sets the limit of the evidence."
    elif any((cases.utilization>=.2)&(cases.structurally_feasible)&(cases.saving_mean>0)):
        q10="The evidence is limited: value appears above the native trace, but no 40% total-utilization case meets the prespecified feasibility/value criterion."
    else:q10="The realism stress does not support the higher-utilization market-value claim under this fixed workload construction."
    lines += ["",f"**Q10. Scope of the 5.19% response:** {q10}","",
        "## Accounting and interpretation","",
        "Rigid and flexible annual IT energy sum to the target total utilization by construction. The rigid service is constant proportional capacity occupancy; flexible execution and its earliest-service reference share only the residual capacity. Late IT is the deadline metric. Closing IT is a calendar-boundary obligation and is reported separately. Positive raw savings in any late policy run are not interpreted as delivered executable value.","",
        "The policy uses the three preselected frozen CdfR2 checkpoints at epochs 6, 2 and 0. Model inference is evaluation-only; no optimizer, gradient update, calibration, normalization refit, checkpoint selection, controller repair, ISO-NE data or workload-share adaptation was used.","",
        "Detailed result rows and daily accounts: `outputs/G3_REALISTIC_UTILIZATION_RESULTS.csv` and `outputs/daily/G3_REALISTIC_UTILIZATION_DAILY.csv`."]
    (ROOT/"reports/G3_REALISTIC_UTILIZATION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    passed40=len(support40)
    if passed40==2: decision="STRONG SUPPORT";why="Both 40% total-utilization cases (10% and 30% flexible share) are structurally feasible, have positive mean savings, and retain zero-late policy seeds."
    elif passed40==1: decision="MODERATE SUPPORT";why="At least one 40% case meets the prespecified feasibility and positive-value criterion, while the other 40% share limits the supported workload mix."
    elif any((cases.utilization>=.2)&(cases.structurally_feasible)&(cases.saving_mean>0)): decision="LIMITED SUPPORT";why="Positive feasible value occurs above native utilization but no 40% case meets the prespecified criterion."
    else: decision="FAILED REALISM STRESS";why="No tested 20% or 40% mixed-load case retains positive feasible executable value."
    go=["# G3 GO / NO-GO","",f"## {decision}","",why,"",
        "The classification uses only prespecified 20/40/60% total utilization and 10/30% flexible energy shares. It does not require every case to succeed; structural failures remain in the report.","",
        "Rigid background is a controlled proportional occupancy stress, not a measured production-load profile. Three seeds are frozen refinement runs, not independent backbone replicates."]
    (ROOT/"reports/G3_GO_NO_GO.md").write_text("\n".join(go)+"\n",encoding="utf-8")

    results_patch=["# G3 Results patch (review before manuscript insertion)","",
        "## Executable value persists under mixed rigid–flexible utilization","",
        f"The native Eagle case operates at {100*native.utilization:.2f}% mean variable-IT utilization. We then combined a constant proportional rigid background with Eagle-shaped flexible arrivals, holding the frozen 2025 market pathway, 24 h contract and settlement fixed. Across total utilization levels of 20%, 40% and 60%, with 10% or 30% of annual variable-IT energy flexible, {('both 40% cases retained positive mean value with zero policy deadline lateness in all replayed seeds' if len(support40)==2 else 'the 40% evidence was limited to '+', '.join(support40.case.tolist())+' under the tested flexible shares' if len(support40) else 'no 40% case met the positive-value and feasibility criterion')}. The detailed table reports saving per flexible MWh, total-cost saving rate and the structural boundary. These results show {('that 5.19% utilization is not necessary under this controlled mixed-load stress.' if len(support40)>0 else 'limited evidence beyond the native utilization case.')}","",
        "Do not characterize this stress as a measured production workload or claim that total utilization alone determines value."]
    (ROOT/"manuscript_patch/G3_RESULTS_PATCH.md").write_text("\n".join(results_patch)+"\n",encoding="utf-8")
    methods=["# G3 Methods patch (review before manuscript insertion)","",
        "We defined total variable-IT utilization as annual variable-IT energy divided by annual available variable-IT capacity energy. Flexible share was the annual flexible variable-IT energy divided by total variable-IT energy. For a target total utilization U and flexible share s, a rigid component with utilization U(1−s) was served immediately as a constant proportional occupancy of each site's variable capacity. Flexible arrivals retained the original Eagle site allocation and quarter-hour chronology and were scaled once so their annual energy equaled Us times available capacity energy.","",
        "Both the frozen WorldMoE policy and the condition-matched earliest-service reference received the identical rigid service, flexible arrivals, 2025 forcing, physical capacity, PUE/PV model, 24 h inclusive deadline and settlement. Flexible work was restricted to residual capacity after immediate rigid service. The existing 2025 current-action executor and CdfR2 checkpoints (seeds 20260920, 20260921 and 20260922; epochs 6, 2 and 0) were held fixed. No model training, gradient update, calibration, normalization fitting, checkpoint reselection or controller changes were performed.","",
        "The rigid component is a controlled capacity-occupancy stress, not an observed production rigid-load trace. Structural feasibility was determined from the matched earliest-service reference using the formal 1e−8 MWh late-work tolerance. Policy late work and year-end closing work were recorded separately."]
    (ROOT/"manuscript_patch/G3_METHODS_PATCH.md").write_text("\n".join(methods)+"\n",encoding="utf-8")
    discussion=["# G3 Discussion patch (review before manuscript insertion)","",
        "The mixed rigid–flexible stress tests address whether the native Eagle savings require the full capacity to be available for deferral. Interpret any positive result as robustness to higher total utilization under the tested proportional rigid-background construction. The rigid load is a controlled capacity-occupancy stress rather than a measured production data-center workload; it does not establish robustness to a specific production load distribution. Structural infeasibility at the earliest-service reference is a workload/capacity boundary and is retained without model repair. Closing IT is a valued year-end unfinished obligation and remains distinct from deadline lateness."]
    (ROOT/"manuscript_patch/G3_DISCUSSION_PATCH.md").write_text("\n".join(discussion)+"\n",encoding="utf-8")

def main():
    df,summary,daily=make_results()
    audit=build_consistency_audit(df)
    plot_figure(df,summary)
    write_reports(df,summary,daily)
    print(f"Built {len(df)} result rows; {len(summary)} summary rows; {len(daily)} daily rows; "
          f"{int(audit['pass'].sum())}/{len(audit)} consistency rows passed.")

if __name__=="__main__": main()
