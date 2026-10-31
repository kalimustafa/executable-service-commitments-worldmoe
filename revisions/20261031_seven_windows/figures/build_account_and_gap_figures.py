"""Source-backed cash-flow waterfall and actual-admission 24 h opportunity."""
from pathlib import Path
from decimal import Decimal, getcontext
from fractions import Fraction
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

getcontext().prec=70
OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
OLD=ROOT/'prior_evidence'
if not OLD.exists():OLD=ROOT.parent/'EnergyEconomics_Revision_20261031'
LEDGER=OLD/'historical_extension/results/historical_settlement.csv'
KEY=ROOT/'certificates/results/key_economic_comparisons.json'
BOUNDS=ROOT/'certificates/results/seven_window_bounds.csv'
C={'purple':'#a34fbe','pink':'#e563a0','lavender':'#e6d5ed',
   'text':'#38223f','axis':'#5d5662','grid':'#eeeaf0'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
 'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':C['axis'],
 'axes.labelcolor':C['text'],'text.color':C['text'],'xtick.color':C['axis'],
 'ytick.color':C['axis'],'savefig.facecolor':'white','axes.linewidth':.8,
 'svg.fonttype':'none','pdf.fonttype':42})

def dfrac(s):
    f=Fraction(s);return Decimal(f.numerator)/Decimal(f.denominator)
def readcsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ('png','svg','pdf'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='png' else 300,
                    bbox_inches='tight',pad_inches=.07)
    plt.close(fig)

ledger=[r for r in readcsv(LEDGER) if int(r['window_h'])==24]
if len(ledger)!=3:raise ValueError('Expected three recovered 24 h cash-flow records')
mean=lambda key:sum(Decimal(r[key]) for r in ledger)/Decimal(3)
gross=mean('gross_avoided_energy_usd');recovery=mean('recovery_energy_usd')
energy=gross-recovery;deviation=mean('execution_change_charge_usd')
curtailment=mean('curtailment_benefit_usd');net=energy-deviation+curtailment
recorded=mean('saving_usd')
if abs(net-recorded)>Decimal('0.000001'):raise ValueError('Cash-flow rows do not reconcile')
stages=[gross,-recovery,energy,-deviation,curtailment,net]
bottom=[0,energy,0,energy-deviation,energy-deviation,0]
heights=[gross,recovery,energy,deviation,curtailment,net]
colors=[C['lavender'],C['pink'],C['lavender'],C['pink'],C['purple'],C['lavender']]
fig,ax=plt.subplots(figsize=(6.8,3.0))
for i,(b,h,color) in enumerate(zip(bottom,heights,colors)):
    ax.bar(i,float(h)/1000,bottom=float(b)/1000,width=.62,color=color,
           edgecolor=C['axis'] if i in (0,2,5) else color,lw=.55,zorder=3)
# Adjacent bridge connectors clarify that the middle energy bar is a subtotal.
for i,level in [(0,gross),(1,energy),(2,energy),(3,energy-deviation),(4,net)]:
    ax.plot([i+.31,i+.69],[float(level)/1000]*2,color=C['axis'],lw=.7,zorder=2)
for i,v in enumerate(stages):
    label=(f'−{abs(v):,.2f}' if v<0 else ('+' if i==4 else '')+f'{v:,.2f}')
    top=float(bottom[i]+heights[i])/1000
    if i==1:
        ax.text(i,float(energy+recovery/2)/1000,label,ha='center',va='center',fontsize=9)
    else:ax.text(i,top+4.5,label,ha='center',va='bottom',fontsize=9,
                 color=C['purple'] if i==4 else C['text'])
ax.set_xticks(range(6),['Gross avoided\nimports','Recovery\nimports','Net energy\nreduction',
                       'Deviation\ncharge','Curtailment\nbenefit','Net operating\nsaving'])
ax.set_ylim(0,195);ax.set_yticks([0,50,100,150],['0','50,000','100,000','150,000'])
ax.set_ylabel('Annual value (US$)')
ax.set_title('24 h operating cash-flow reconciliation',loc='left',pad=10)
ax.grid(axis='y',color=C['grid'],lw=.7,zorder=0)
fig.subplots_adjust(left=.12,right=.99,top=.86,bottom=.25)
save(fig,'figure_24h_cashflow_reconciliation')

key=json.loads(KEY.read_text(encoding='utf-8'))
bound=next(r for r in readcsv(BOUNDS) if int(r['window_hours'])==24)
policy=dfrac(key['mean_worldmoe_saving_exact'])
lower=dfrac(bound['combined_floor_exact']);upper=dfrac(bound['combined_ceiling_exact'])
glo=dfrac(key['24h_remaining_opportunity_lower_exact'])
ghi=dfrac(key['24h_remaining_opportunity_upper_exact'])
glo_print=key['24h_remaining_opportunity_lower_down_cents']
ghi_print=key['24h_remaining_opportunity_upper_up_cents']
rlo_print=key['24h_capture_percent_lower_down_2dp'];rhi_print=key['24h_capture_percent_upper_up_2dp']
fig,axes=plt.subplots(1,2,figsize=(7.1,3.35),gridspec_kw={'width_ratios':[1.16,1]})
ax=axes[0]
ax.barh([1],[float(policy)/1000],height=.25,color=C['purple'],zorder=3)
ax.barh([0],[float(lower)/1000],height=.25,color=C['lavender'],zorder=3)
ax.barh([0],[float(upper-lower)/1000],left=float(lower)/1000,height=.25,color=C['pink'],zorder=4)
ax.vlines([float(lower)/1000,float(upper)/1000],-.18,.18,colors=C['pink'],lw=1.2,zorder=5)
ax.text(float(policy)/1000+.8,1,'105.27',va='center',fontsize=9,fontweight='bold')
ax.text(132,.24,'132.38–132.60',ha='right',va='bottom',fontsize=8.5,fontweight='bold')
ax.set_yticks([1,0],['WorldMoE','24 h opportunity'])
ax.set_ylim(-.65,1.5);ax.set_xlim(0,165);ax.set_xticks([0,40,80,120,160])
ax.set_xlabel('Annual saving (US$ thousand)')
ax.set_title('(a) Policy value and opportunity',loc='left',pad=11)
ax.grid(axis='x',color=C['grid'],lw=.7,zorder=0);ax.spines['left'].set_visible(False)
ax.tick_params(axis='y',length=0)
ax.text(0,-.5,'Actual-admission certificate interval',fontsize=7.5,color=C['axis'])
ax=axes[1]
ax.hlines(0,float(glo)/1000,float(ghi)/1000,color=C['pink'],lw=5,zorder=3)
ax.scatter([float(glo)/1000,float(ghi)/1000],[0,0],s=35,color=C['pink'],
           edgecolors='white',lw=.6,zorder=4)
ax.text(17.5,.19,f'US${Decimal(glo_print):,.2f}–{Decimal(ghi_print):,.2f}',
        ha='center',va='bottom',fontsize=9.2,fontweight='bold')
ax.set_xlim(0,35);ax.set_xticks([0,10,20,30]);ax.set_ylim(-.38,.53);ax.set_yticks([])
ax.set_xlabel('Remaining opportunity (US$ thousand)')
ax.set_title('(b) Remaining 24 h opportunity',loc='left',pad=11)
ax.spines['left'].set_visible(False);ax.grid(axis='x',color=C['grid'],lw=.7,zorder=0)
ax.text(.04,.17,'Opportunity captured',transform=ax.transAxes,fontsize=8.5)
ax.text(.04,.04,f'{rlo_print}%–{rhi_print}%',transform=ax.transAxes,
        fontsize=13,fontweight='bold',color=C['purple'])
fig.subplots_adjust(left=.14,right=.99,bottom=.22,top=.86,wspace=.38)
fig.text(.14,.025,'Exact repair; matched admissions and account. Bounds rounded outward.',fontsize=8,color=C['axis'])
save(fig,'figure3_remaining_opportunity_actual_admissions')

record={'palette':C,'cashflow_source':str(LEDGER),
 'cashflow_means_usd':{'gross_avoided_import':str(gross),'recovery_import_cost':str(recovery),
 'net_energy_reduction':str(energy),'import_deviation_charge':str(deviation),
 'pv_curtailment_benefit':str(curtailment),'net_saving_from_components':str(net),
 'mean_recorded_saving':str(recorded),'reconciliation_residual':str(net-recorded)},
 'actual_admission_gap':key,'24h_certificate_source_row':bound,
 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LEDGER,KEY,BOUNDS]},
 'scope':'Historical three-seed 24 h reconciled cash flows and actual-admission exact certificates. '
         'No additional tenant compensation or implementation cost is included. '
         'The earlier uniform-admission certificate remains in the historical package, not in these main figures.'}
(OUT/'account_and_gap_figure_data.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Created exact-source waterfall and actual-admission opportunity figure.')
