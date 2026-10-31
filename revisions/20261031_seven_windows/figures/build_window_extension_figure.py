"""Render seven-window evidence, tables and paired marginal gains.

Input contracts are documented in figure_inputs.txt. This program never fills
missing results or estimates certificate endpoints from policy performance.
"""
from pathlib import Path
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
from fractions import Fraction
import argparse
import csv
import hashlib
import json
import math

WINDOWS=(8,12,16,20,24,36,48)
getcontext().prec=70
ORIGINS=('recovered_historical','new_20261031')
PALETTE={'purple':'#a34fbe','pink':'#e563a0','lightpink':'#edacd0',
         'lavender':'#e6d5ed','text':'#38223f','axis':'#5d5662','grid':'#eeeaf0'}

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def amount(row,key):
    try:
        raw=str(row[key]); f=Fraction(raw)
        value=Decimal(f.numerator)/Decimal(f.denominator)
    except Exception as exc: raise ValueError(f'Missing/invalid {key}: {row}') from exc
    if not value.is_finite(): raise ValueError(f'Nonfinite {key}: {row}')
    return value

def service_ok(row):
    value=str(row.get('service_feasible','')).lower()
    if value not in ('true','false','1','0'):
        raise ValueError('Every policy row needs explicit service_feasible true/false')
    return value in ('true','1')

def cents(value,direction=None):
    return str(value.quantize(Decimal('.01'),rounding=direction)) if direction else f'{value:.2f}'

def save_csv(path, records):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--certificates',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,help='Optional separate 24 h rerun, three matched seeds')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    policy_rows,certificate_rows=rows(args.policy),rows(args.certificates)
    by_window={w:{} for w in WINDOWS}; certs={}
    for row in policy_rows:
        w=int(row['window_h']); seed=str(row['seed'])
        if w not in by_window or seed in by_window[w]:
            raise ValueError(f'Unexpected/duplicate window-seed row: {w}, {seed}')
        if row['evidence_origin'] not in ORIGINS or not row.get('source_path'):
            raise ValueError('Every policy row needs an explicit allowed evidence origin and source_path')
        service_ok(row); amount(row,'late_it_mwh')
        by_window[w][seed]=(amount(row,'annual_saving_usd'),row)
    seed_sets={tuple(sorted(by_window[w])) for w in WINDOWS}
    if len(seed_sets)!=1 or len(next(iter(seed_sets)))!=3:
        raise ValueError('All seven windows require exactly the same three seeds; no missing-data plot is produced')
    seeds=next(iter(seed_sets))
    for row in certificate_rows:
        w=int(row['window_h'])
        if w not in by_window or w in certs or not row.get('source_path'):
            raise ValueError(f'Unexpected/duplicate certificate or missing source: {row}')
        lo,hi=amount(row,'pi_lower_usd'),amount(row,'pi_upper_usd')
        if lo>hi:raise ValueError(f'Reversed certificate interval: {w}')
        certs[w]=(lo,hi,row)
    if set(certs)!=set(WINDOWS):raise ValueError('All seven PI intervals are required; missing endpoints are not inferred')
    summaries=[]; marginal=[]
    for w in WINDOWS:
        values=[by_window[w][s][0] for s in seeds]
        origins={by_window[w][s][1]['evidence_origin'] for s in seeds}
        if len(origins)!=1:raise ValueError(f'Mixed provenance inside {w} h three-seed group')
        mean=sum(values)/Decimal(3);lo,hi,_=certs[w]
        feasible=all(service_ok(by_window[w][s][1]) for s in seeds)
        late=[amount(by_window[w][s][1],'late_it_mwh') for s in seeds]
        summaries.append({'window_h':w,'evidence_origin':next(iter(origins)),
            'seed_count':3,'policy_mean_usd':str(mean),'policy_seed_min_usd':str(min(values)),
            'policy_seed_max_usd':str(max(values)),'service_feasible':feasible,
            'mean_late_it_mwh':str(sum(late)/Decimal(3)),'maximum_late_it_mwh':str(max(late)),
            'policy_value_status':'contract-feasible saving' if feasible else 'cash-only service failure',
            'pi_lower_usd':str(lo),'pi_upper_usd':str(hi),
            'capture_percent_lower':str(mean/hi*100) if feasible else '',
            'capture_percent_upper':str(mean/lo*100) if feasible else ''})
    for wa,wb in zip(WINDOWS[:-1],WINDOWS[1:]):
        hours=Decimal(wb-wa)
        gains=[(by_window[wb][s][0]-by_window[wa][s][0])/hours for s in seeds]
        valid=all(service_ok(by_window[w][s][1]) for w in (wa,wb) for s in seeds)
        # Nested feasible sets make the optimum nondecreasing; intersect the
        # difference interval with [0,infinity), without imposing monotonicity
        # on the realized policy trajectory.
        rawlo=certs[wb][0]-certs[wa][1]
        rawhi=certs[wb][1]-certs[wa][0]
        if rawhi<0:raise ValueError(f'Certificates conflict with nested-rights monotonicity: {wa}->{wb}')
        marginal.append({'from_h':wa,'to_h':wb,'added_hours':int(hours),
            'policy_gain_valid':valid,
            'policy_paired_mean_usd_per_h':str(sum(gains)/Decimal(3)) if valid else '',
            'policy_paired_seed_min_usd_per_h':str(min(gains)) if valid else '',
            'policy_paired_seed_max_usd_per_h':str(max(gains)) if valid else '',
            'pi_increment_lower_usd_per_h':str(max(Decimal(0),rawlo)/hours),
            'pi_increment_upper_usd_per_h':str(rawhi/hours),
            'scope':'Historical paired difference quotient; not a derivative or future-year expected marginal value'})
    anchors=[]
    if args.anchor:
        anchor_rows=rows(args.anchor)
        if len(anchor_rows)!=3 or {str(r['seed']) for r in anchor_rows}!=set(seeds) or any(int(r['window_h'])!=24 for r in anchor_rows):
            raise ValueError('Anchor must contain precisely the same three seeds at 24 h')
        for row in anchor_rows:
            seed=str(row['seed']);value=amount(row,'annual_saving_usd')
            anchors.append({'seed':seed,'rerun_saving_usd':str(value),
                'primary_24h_saving_usd':str(by_window[24][seed][0]),
                'difference_usd':str(value-by_window[24][seed][0]),
                'source_path':row['source_path']})
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    save_csv(out/'seven_window_summary.csv',summaries)
    save_csv(out/'adjacent_window_marginal_values.csv',marginal)
    if anchors:save_csv(out/'anchor_24h_comparison.csv',anchors)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.edgecolor':PALETTE['axis'],'axes.labelcolor':PALETTE['text'],
        'text.color':PALETTE['text'],'xtick.color':PALETTE['axis'],
        'ytick.color':PALETTE['axis'],'savefig.facecolor':'white',
        'axes.linewidth':.8,'svg.fonttype':'none','pdf.fonttype':42})
    # Actual unequal window spacing is retained. No interpolation is asserted
    # between evaluated points: the thin lines merely connect plotted values.
    fig,ax=plt.subplots(2,1,figsize=(7.1,5.7),gridspec_kw={'height_ratios':[1.5,1]})
    x=np.array(WINDOWS,dtype=float)
    mean=np.array([float(r['policy_mean_usd'])/1000 for r in summaries])
    plow=np.array([float(r['policy_seed_min_usd'])/1000 for r in summaries])
    phigh=np.array([float(r['policy_seed_max_usd'])/1000 for r in summaries])
    lower=np.array([float(r['pi_lower_usd'])/1000 for r in summaries])
    upper=np.array([float(r['pi_upper_usd'])/1000 for r in summaries])
    ax[0].vlines(x,lower,upper,color=PALETTE['pink'],lw=5,zorder=3,alpha=.65)
    ax[0].plot(x,lower,ls=':',color=PALETTE['pink'],lw=.85,zorder=2)
    ax[0].plot(x,upper,ls=':',color=PALETTE['pink'],lw=.85,zorder=2)
    feasible=np.array([r['service_feasible'] for r in summaries],dtype=bool)
    ax[0].plot(x,np.where(feasible,mean,np.nan),color=PALETTE['purple'],lw=1.25,zorder=4)
    ax[0].errorbar(x[feasible],mean[feasible],yerr=[(mean-plow)[feasible],(phigh-mean)[feasible]],fmt='none',ecolor=PALETTE['purple'],
                   capsize=3,lw=1,zorder=5)
    for i,row in enumerate(summaries):
        if not row['service_feasible']:
            fail_values=[float(by_window[row['window_h']][s][0])/1000 for s in seeds]
            ax[0].scatter([x[i]-.24,x[i],x[i]+.24],fail_values,s=42,marker='x',
                          color=PALETTE['pink'],linewidths=1.2,zorder=7)
            ax[0].annotate(f"{row['window_h']} h service failure",(x[i],mean[i]),
                xytext=(x[i]+3,mean[i]-13),fontsize=8,color=PALETTE['pink'],
                arrowprops={'arrowstyle':'-','color':PALETTE['pink'],'lw':.7})
            continue
        hist=row['evidence_origin']=='recovered_historical'
        ax[0].scatter([x[i]],[mean[i]],s=44,marker='o' if hist else 'D',
                      facecolors='white' if hist else PALETTE['purple'],
                      edgecolors=PALETTE['purple'],linewidths=1.1,zorder=6)
    if anchors:
        aval=[float(r['rerun_saving_usd'])/1000 for r in anchors]
        # The rerun is an independently stored anchor, not an eighth window.
        ax[0].scatter([24],[sum(aval)/3],s=95,marker='+',color=PALETTE['text'],
                      linewidths=1.2,zorder=7)
    handles=[Line2D([],[],marker='o',mfc='white',mec=PALETTE['purple'],color=PALETTE['purple'],
                    label='Recovered historical policy'),
             Line2D([],[],marker='D',mfc=PALETTE['purple'],mec=PALETTE['purple'],color=PALETTE['purple'],
                    label='New 31 Oct policy evaluation'),
             Line2D([],[],color=PALETTE['pink'],lw=4,alpha=.65,label='Certified PI interval')]
    if not feasible.all():handles.append(Line2D([],[],marker='x',color=PALETTE['pink'],ls='none',label='Cash only; service failure'))
    if anchors:handles.append(Line2D([],[],marker='+',color=PALETTE['text'],ls='none',label='24 h rerun anchor'))
    ax[0].legend(handles=handles,loc='upper left',frameon=False,ncol=2,fontsize=7.3,
                  handlelength=1.8,columnspacing=1.3)
    ax[0].set_title('(a) Operating policy and contract opportunity',loc='left',pad=10)
    ax[0].set_ylabel('Annual saving (US$ thousand)')
    ymin=min(0,float(min(plow.min(),lower.min()))*.95)
    ymax=max(float(phigh.max()),float(upper.max()))
    ax[0].set_ylim(ymin,ymax*1.25 if ymax>0 else 1)
    ax[0].set_xlim(6,50);ax[0].set_xticks(WINDOWS)
    ax[0].set_xlabel('Contract window (h)')
    ax[0].grid(axis='y',color=PALETTE['grid'],lw=.7,zorder=0)
    mids=np.array([(r['from_h']+r['to_h'])/2 for r in marginal])
    widths=np.array([r['added_hours']*.62 for r in marginal])
    valid=np.array([r['policy_gain_valid'] for r in marginal],dtype=bool)
    number=lambda r,k:float(r[k])/1000 if r[k] else np.nan
    mg=np.array([number(r,'policy_paired_mean_usd_per_h') for r in marginal])
    ml=np.array([number(r,'policy_paired_seed_min_usd_per_h') for r in marginal])
    mu=np.array([number(r,'policy_paired_seed_max_usd_per_h') for r in marginal])
    px=mids-widths*.16;qx=mids+widths*.39
    ax[1].bar(px[valid],mg[valid],width=widths[valid]*.63,color=PALETTE['lavender'],
              edgecolor=PALETTE['purple'],lw=.8,zorder=3,label='Policy gain')
    ax[1].errorbar(px[valid],mg[valid],yerr=[(mg-ml)[valid],(mu-mg)[valid]],fmt='none',
                  ecolor=PALETTE['purple'],capsize=3,lw=1,zorder=4)
    pi_lo=np.array([float(r['pi_increment_lower_usd_per_h'])/1000 for r in marginal])
    pi_hi=np.array([float(r['pi_increment_upper_usd_per_h'])/1000 for r in marginal])
    ax[1].vlines(qx,pi_lo,pi_hi,color=PALETTE['pink'],lw=4,zorder=5,label='PI marginal interval')
    for j,is_valid in enumerate(valid):
        if not is_valid:ax[1].text(px[j],.1,'NA',ha='center',va='bottom',fontsize=8,color=PALETTE['axis'])
    ax[1].axhline(0,color=PALETTE['axis'],lw=.8,zorder=2)
    ax[1].set_title('(b) Adjacent-window gain per added hour',loc='left',pad=10)
    ax[1].set_ylabel('Annual gain per added hour\n(US$ thousand / h)')
    ax[1].set_xlim(6,50)
    ax[1].set_xticks(mids,[f"{r['from_h']}–{r['to_h']}" for r in marginal])
    ax[1].set_xlabel('Contract-window comparison (h)')
    ax[1].grid(axis='y',color=PALETTE['grid'],lw=.7,zorder=0)
    ax[1].legend(loc='upper right',frameon=False,fontsize=8)
    fig.subplots_adjust(left=.115,right=.98,top=.95,bottom=.13,hspace=.52)
    fig.text(.115,.025,'Whiskers: three-seed range. PI: certified bounds.\n'
             '8 h: cash only, failed service. 36/48 h: declared continuation.',
             fontsize=8,color=PALETTE['axis'])
    for ext in ('png','svg','pdf'):
        fig.savefig(out/f'figure_seven_window_contract_value.{ext}',dpi=600 if ext=='png' else 300,
                    bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
    # Tables use outward rounding of certified endpoints. Means/ranges and
    # policy marginal values are descriptive decimal summaries.
    table=['Table X. Seven service windows and historical-path value.','',
       '| Window (h) | Policy/cash mean (US$) | Three-seed range (US$) | Service | PI interval (US$) | Capture (%) | Evidence |',
       '|---:|---:|---:|---|---:|---:|---|']
    for r in summaries:
        lo,hi=Decimal(r['pi_lower_usd']),Decimal(r['pi_upper_usd'])
        capture=f"{cents(Decimal(r['capture_percent_lower']),ROUND_FLOOR)}–{cents(Decimal(r['capture_percent_upper']),ROUND_CEILING)}" if r['service_feasible'] else '—'
        service='Feasible' if r['service_feasible'] else f"Failed; {Decimal(r['mean_late_it_mwh']):.3f} IT MWh late"
        table.append(f"| {r['window_h']} | {cents(Decimal(r['policy_mean_usd']))} | "
          f"{cents(Decimal(r['policy_seed_min_usd']))}–{cents(Decimal(r['policy_seed_max_usd']))} | {service} | "
          f"[{cents(lo,ROUND_FLOOR)}, {cents(hi,ROUND_CEILING)}] | {capture} | {r['evidence_origin']} |")
    table += ['','All values use the same historical calendar, admitted workload and financial account. '
              'Policy ranges describe three matched seeds, not future-year confidence intervals. '
              'Certificate bounds are rounded outward to cents. '
              'New evaluations use frozen checkpoints; 36/48 h results include the declared continuation beyond the 24 h neural horizon.']
    (out/'seven_window_table.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.policy,args.certificates]+([args.anchor] if args.anchor else [])}
    provenance={'sources_sha256':sources,'windows_h':WINDOWS,'seeds':seeds,'palette':PALETTE,
        'summary':summaries,'marginals':marginal,'anchor':anchors,
        'scope':'Historical-path policy response and independently certified PI opportunities; no ex-ante, routing-only, or fully neural long-horizon claim.'}
    (out/'figure_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Rendered seven-window figure and source-backed tables.')

if __name__=='__main__': main()
