"""Canonical publication figures. Existing results only; no fitting or replay."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from theme import MODEL_COLORS as COLORS,INK,GRID,NO_FLEX

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'source_data';F=ROOT/'figures'
W=COLORS['WorldMoE'];REF=NO_FLEX
LABELS={'World-ME':'Previous WorldMoE','WorldMoE':'WorldMoE','Legacy-price':'Price-only head','World-price':'Previous price head',
        'LGB-Full-ME':'LightGBM full','LGB-Recent-ME':'LightGBM recent','LGB-NoState-ME':'LightGBM no-state','GlobalCal-ME':'Global calibration',
        'DA-myopic':'Published DA myopic','DA-persistent':'Published DA persistent','RT-lag':'Lagged RT','No-flex':'Immediate execution'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':7.5,
 'xtick.labelsize':6.5,'ytick.labelsize':6.2,'legend.fontsize':6.2,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.6,'axes.edgecolor':'#B895BC','text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,'svg.fonttype':'none',
 'savefig.facecolor':'white'})
MANIFEST=[]
def read(n):return pd.read_csv(D/n,encoding='utf-8-sig')
def lab(m):return LABELS.get(m,m)
def col(m):return COLORS.get(m,REF)
def panel(ax,l,t,grid='y'):
    ax.set_title(f'({l}) {t}',loc='left',fontweight='bold',pad=8,fontsize=8)
    ax.grid(axis=grid,color=GRID,lw=.5);ax.set_axisbelow(True)
def save(fig,name,sources,scope,extended=False):
    out=F/('extended_data' if extended else 'main');out.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=600 if ext=='png' else None,bbox_inches='tight',pad_inches=.06)
    plt.close(fig);MANIFEST.append(dict(figure=name,source_script='figure_source/final_figures.py',sources=sources,comparison_scope=scope,png_dpi=600))
def axes22(h=5.9):
    fig,ax=plt.subplots(2,2,figsize=(7.2,h));fig.subplots_adjust(left=.13,right=.98,bottom=.10,top=.94,wspace=.60,hspace=.70);return fig,ax.flat
def shortbar(ax,names,values,colors=None):
    yy=np.arange(len(values));ax.barh(yy,values,color=colors if colors is not None else W,height=.63)
    ax.set_yticks(yy,names);ax.invert_yaxis();ax.grid(axis='x',color=GRID,lw=.5)

if __name__=='__main__':
    from compact_figures import main
    main()
