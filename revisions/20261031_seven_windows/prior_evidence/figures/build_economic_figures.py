"""Reproduce revised economic figures in the original manuscript palette.

Inputs: preserved 2026-10-31 economic calculations and displayed conservative
24-hour endpoints. No underlying historical experiment is re-estimated here.
"""
from pathlib import Path
from decimal import Decimal
import json
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
SOURCE = ROOT / 'source_package/verification_20261031/economics/economic_value_results.json'
DATA = json.loads(SOURCE.read_text(encoding='utf-8'))
P = {'purple':'#a34fbe', 'pink':'#e563a0', 'lightpink':'#edacd0',
     'lavender':'#e6d5ed', 'text':'#38223f', 'axis':'#5d5662',
     'grid':'#eeeaf0', 'white':'#ffffff'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
    'axes.titlesize':10, 'axes.labelsize':9, 'xtick.labelsize':8,
    'ytick.labelsize':8, 'axes.spines.top':False, 'axes.spines.right':False,
    'axes.edgecolor':P['axis'], 'axes.labelcolor':P['text'],
    'text.color':P['text'], 'xtick.color':P['axis'], 'ytick.color':P['axis'],
    'savefig.facecolor':'white', 'axes.linewidth':.8,
    'svg.fonttype':'none', 'pdf.fonttype':42})

def save(fig, name):
    for ext in ('png','svg','pdf'):
        fig.savefig(OUT / (name+'.'+ext), dpi=600 if ext=='png' else 300,
                    bbox_inches='tight', pad_inches=.07)
    plt.close(fig)

# Figure 2: retain historical data and analytical fee scenarios.
fig, axes = plt.subplots(1,2,figsize=(7.1,3.25),
                         gridspec_kw={'width_ratios':[.92,1.13]})
rights=[12,16,24]
means=np.array([float(DATA['contract_means_usd'][str(w)])/1000 for w in rights])
colors=[P['lavender'],P['lightpink'],P['purple']]
axes[0].bar(range(3),means,width=.57,color=colors,zorder=3)
for i,y in enumerate(means):
    axes[0].text(i,y+2,f'{y:.2f}',ha='center',va='bottom',fontweight='bold')
axes[0].set_xticks(range(3),['12 h','16 h','24 h'])
axes[0].set_ylim(0,120)
axes[0].set_yticks([0,30,60,90,120])
axes[0].set_xlabel('Contract window')
axes[0].set_ylabel('Annual saving (US$ thousand)')
axes[0].set_title('(a) Historical operating value',loc='left',pad=11)
axes[0].grid(axis='y',color=P['grid'],lw=.7,zorder=0)
cost=np.linspace(0,36,181)
for row,col,style in zip(DATA['paired_upgrade_summary'],
                      [P['lightpink'],P['pink'],P['purple']],[':','--','-']):
    mean=float(row['mean_increment_usd'])/1000
    lo=float(row['minimum_seed_increment_usd'])/1000
    hi=float(row['maximum_seed_increment_usd'])/1000
    axes[1].plot(cost,mean-cost,color=col,lw=2,ls=style,
                 label=f"{row['from_h']} to {row['to_h']} h",zorder=4)
    axes[1].fill_between(cost,lo-cost,hi-cost,color=col,alpha=.2,lw=0,zorder=3)
    axes[1].scatter([mean],[0],s=20,color=col,zorder=5)
axes[1].axhline(0,color=P['axis'],lw=.8,zorder=2)
axes[1].set_xlim(0,36)
axes[1].set_ylim(-24,36)
axes[1].set_xticks([0,10,20,30])
axes[1].set_yticks([-20,-10,0,10,20,30])
axes[1].set_xlabel('Assumed added compensation and costs\n(US$ thousand)')
axes[1].set_ylabel('Incremental net saving (US$ thousand)')
axes[1].set_title('(b) Historical cost sensitivity',loc='left',pad=11)
axes[1].grid(color=P['grid'],lw=.7,zorder=0)
axes[1].legend(loc='upper right',frameon=False,fontsize=8,handlelength=2.4)
fig.subplots_adjust(left=.09,right=.985,bottom=.22,top=.88,wspace=.46)
save(fig,'figure2_historical_economic_value')

# Figure 3: interval arithmetic, never a statistical confidence interval.
policy=Decimal(DATA['contract_means_usd']['24'])
lower,upper=Decimal('130931.31'),Decimal('134437.85')
glo,ghi=lower-policy,upper-policy
rlo,rhi=policy/upper*100,policy/lower*100
fig, axes = plt.subplots(1,2,figsize=(7.1,3.35),
                         gridspec_kw={'width_ratios':[1.16,1]})
ax=axes[0]
ax.barh([1],[float(policy)/1000],height=.25,color=P['purple'],zorder=3)
ax.barh([0],[float(lower)/1000],height=.25,color=P['lavender'],zorder=3)
ax.barh([0],[float(upper-lower)/1000],left=float(lower)/1000,
        height=.25,color=P['pink'],zorder=4)
ax.vlines([float(lower)/1000,float(upper)/1000],-.18,.18,
          colors=P['pink'],lw=1.2,zorder=5)
ax.text(float(policy)/1000+.8,1,'105.27',va='center',fontsize=9,fontweight='bold')
ax.text(130,.24,'130.93–134.44',ha='right',va='bottom',fontsize=8.5,fontweight='bold')
ax.set_yticks([1,0],['WorldMoE','24 h opportunity'])
ax.set_ylim(-.65,1.5)
ax.set_xlim(0,165)
ax.set_xticks([0,40,80,120,160])
ax.set_xlabel('Annual saving (US$ thousand)')
ax.set_title('(a) Policy value and opportunity',loc='left',pad=11)
ax.grid(axis='x',color=P['grid'],lw=.7,zorder=0)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y',length=0)
ax.text(0,-.5,'Certified interval, not a confidence interval',fontsize=7.1,color=P['axis'])

ax=axes[1]
ax.hlines(0,float(glo)/1000,float(ghi)/1000,color=P['pink'],lw=5,zorder=3)
ax.scatter([float(glo)/1000,float(ghi)/1000],[0,0],s=35,
           color=P['pink'],edgecolors='white',lw=.6,zorder=4)
ax.text(17.5,.16,
        f'≈US${float(glo):,.0f}–{float(ghi):,.0f}',ha='center',va='bottom',
        fontsize=10,fontweight='bold')
ax.set_xlim(0,35)
ax.set_xticks([0,10,20,30])
ax.set_ylim(-.38,.53)
ax.set_yticks([])
ax.set_xlabel('Remaining opportunity (US$ thousand)')
ax.set_title('(b) Remaining 24 h opportunity',loc='left',pad=11)
ax.spines['left'].set_visible(False)
ax.grid(axis='x',color=P['grid'],lw=.7,zorder=0)
ax.text(.04,.17,'Opportunity captured',transform=ax.transAxes,fontsize=8.5)
ax.text(.04,.04,f'≈{float(rlo):.2f}%–{float(rhi):.2f}%',transform=ax.transAxes,
        fontsize=13,fontweight='bold',color=P['purple'])
fig.subplots_adjust(left=.14,right=.99,bottom=.22,top=.86,wspace=.38)
fig.text(.14,.025,'Matched admissions, original 24 h service constraints and common financial account;\nexact numerical repair.',
         fontsize=7.2,color=P['axis'])
save(fig,'figure3_remaining_opportunity')

records={'palette':P,'source_economics':str(SOURCE),
 'values':{'policy_saving_usd':str(policy),'24h_lower_usd':str(lower),
           '24h_upper_usd':str(upper),'gap_lower_usd':str(glo),
           'gap_upper_usd':str(ghi),'capture_lower_percent':str(rlo),
           'capture_upper_percent':str(rhi)},
 'scope':'Historical calendar; matched admissions, original 24 h service constraints and common financial account; exact numerical repair. '
         'All nine recovered historical trajectories have exact service/account repairs; the 24 h mean enclosure is [105271.061841632,105271.061841633] USD and the maximum repair cost is below 1.66e-8 USD. '
         'Graph labels use nearest whole dollars or two decimal percentages, not outward-rounded certificates. '
         'The opportunity interval is deterministic and not statistical. Historical refinement bands are not confidence intervals.',
 'figure2':'Historical paired policy results and hypothetical additional-cost sensitivity; values unchanged.',
 'figure3':'Difference and ratio derived from the published 24 h opportunity endpoints and WorldMoE historical mean.'}
(OUT/'economic_figure_data.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Created Figure 2 and Figure 3 in PNG, SVG and PDF.')
