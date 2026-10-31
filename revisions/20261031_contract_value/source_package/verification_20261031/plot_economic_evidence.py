from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/contract_value_matplotlib')
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
data=json.loads((ROOT/'economics/economic_value_results.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
 'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#5b6570','axes.labelcolor':'#182a3a','text.color':'#182a3a','savefig.facecolor':'white'})
fig,ax=plt.subplots(1,2,figsize=(7.1,3.55),gridspec_kw={'width_ratios':[0.94,1.1]})
rights=[12,16,24]
means=np.array([float(data['contract_means_usd'][str(x)])/1000 for x in rights])
colors=['#7793a9','#3c7395','#173e60']
ax[0].bar(np.arange(3),means,width=.6,color=colors,zorder=3)
for i,(right,y) in enumerate(zip(rights,means)):
    ax[0].text(i,y+2,f'{y:.2f}',ha='center',va='bottom',fontsize=10,fontweight='bold')
ax[0].set_xticks(np.arange(3),['12 h','16 h','24 h'])
ax[0].set_ylim(0,120)
ax[0].set_ylabel('Annual operating saving\n(US$ thousand)')
ax[0].set_xlabel('Contract window')
ax[0].set_title('(a) Historical operating saving',loc='left',pad=12)
ax[0].grid(axis='y',color='#dce2e7',lw=.6,zorder=0)
cost=np.linspace(0,36,181)
for r,color in zip(data['paired_upgrade_summary'],['#76929b','#d27b39','#173e60']):
    y=float(r['mean_increment_usd'])/1000
    lo=float(r['minimum_seed_increment_usd'])/1000
    hi=float(r['maximum_seed_increment_usd'])/1000
    label=f"{r['from_h']} to {r['to_h']} h"
    ax[1].plot(cost,y-cost,color=color,lw=1.8,label=label,zorder=4)
    ax[1].fill_between(cost,lo-cost,hi-cost,color=color,alpha=.15,lw=0,zorder=3)
    ax[1].scatter([y],[0],s=28,color=color,zorder=5)
ax[1].axhline(0,color='#555e66',lw=.85,zorder=2)
ax[1].set_ylim(-24,36)
ax[1].set_xlim(0,36)
ax[1].set_xticks([0,10,20,30])
ax[1].set_xlabel('Assumed compensation and costs\n(US$ thousand)')
ax[1].set_ylabel('Incremental net saving\n(US$ thousand)')
ax[1].set_title('(b) Net value after added costs',loc='left',pad=12)
ax[1].grid(color='#dce2e7',lw=.6,zorder=0)
ax[1].legend(loc='upper right',frameon=False,fontsize=9,handlelength=1.8)
fig.subplots_adjust(left=.095,right=.985,bottom=.22,top=.88,wspace=.46)
fig.savefig(OUT/'contract_value_and_fee_sensitivity.png',dpi=400,bbox_inches='tight',pad_inches=.08)
fig.savefig(OUT/'contract_value_and_fee_sensitivity.svg',bbox_inches='tight',pad_inches=.08)
plt.close(fig)
print(OUT/'contract_value_and_fee_sensitivity.png')
