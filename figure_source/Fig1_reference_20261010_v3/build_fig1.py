"""WorldMoE-centred scientific architecture; inputs branch into the model.

All shapes are deterministic Python vectors. Folds denote structural depth,
not numerical values. Tables and curves are archived research replay results.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
P=json.loads((HERE/'figure_inputs.json').read_text(encoding='utf-8'))
GEO=json.loads((HERE/'northeast_states.geojson').read_text(encoding='utf-8'))
WINDOWS=pd.read_csv(HERE/'service_windows.csv')
MODELS=pd.read_csv(HERE/'comparison_models.csv').set_index('model_key')
CURVES=pd.read_csv(HERE/'comparison_diurnal_imports.csv')
C=['#B6B3D6','#CFCCE3','#D5D3DE','#D5D1D1','#F6DFD6','#F8B2A2','#F1837A','#E9687A']
INK='#3E3C43'; MUTED='#6B6872'; GRID='#E8E6EB'
W,H=22.,12.3
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
    'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(W,H),facecolor='white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(0,H));ax.axis('off')

def text(x,y,s,size=12,bold=False,ha='left',color=INK,**kw):
    return ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',
                   ha=ha,va='center',color=color,zorder=30,**kw)
def subaxis(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
def route(points,arrow=True,lw=.95,dashed=False):
    if len(points)>2:
        xx,yy=zip(*points[:-1]);ax.plot(xx,yy,lw=lw,color=INK,zorder=20,
                                      ls=(0,(3,2)) if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>' if arrow else '-',
        mutation_scale=11,color=INK,lw=lw,shrinkA=0,shrinkB=0,zorder=20,
        linestyle=(0,(3,2)) if dashed else '-'))
def card(x,y,w,h,fc='white'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.05',
        fc=fc,ec=INK,lw=.9,zorder=1))
def sheet(x,y,w,h,fc=C[1],depth=.08):
    ax.add_patch(Polygon([(x,y+h),(x+w,y+h),(x+w+depth,y+h+depth),(x+depth,y+h+depth)],
        fc='white',ec=INK,lw=.55,zorder=4))
    ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+depth),(x+w+depth,y+h+depth),(x+w,y+h)],
        fc=C[0],ec=INK,lw=.55,zorder=4))
    ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=INK,lw=.7,zorder=5))

text(.35,11.78,'WorldMoE for executable computing flexibility',23,True)

# All left-hand content is a set of model inputs, not serial processing stages.
from portfolio_inputs import draw_portfolio_inputs
inputs=draw_portfolio_inputs(fig,ax,.35,1.35,3.8,9.45,P,GEO)
from worldmoe_reference_core import draw_worldmoe_core
core=draw_worldmoe_core(ax,4.85,1.10,11.1,9.9)

for key,dest in [('observations_port',core['data_input_port']),('queue_port',core['queue_input_port'])]:
    start=inputs[key]
    route([start,(4.52,start[1]),(4.52,dest[1]),dest])

# The external executor translates slot-cost evidence into one current action.
# It is not a neural multi-step dispatch output.
ex,ey,ew,eh=16.38,4.48,1.38,3.64
card(ex,ey,ew,eh,fc='#F8F7FA')
text(ex+ew/2,ey+eh-.35,'Execution',13.7,True,ha='center')
text(ex+ew/2,ey+eh-.67,'15 min',12.8,ha='center')
sheet(ex+.24,ey+2.07,.83,.50,C[1],.055)
for k in range(3):
    ax.add_patch(Rectangle((ex+.34+k*.22,ey+2.22),.13,.14,fc=C[7] if k==0 else C[2],ec=INK,lw=.4,zorder=8))
route([(ex+.68,ey+2.06),(ex+.68,ey+1.77)],lw=.8)
ax.add_patch(Polygon([(ex+.23,ey+1.74),(ex+1.12,ey+1.74),(ex+.86,ey+1.28),(ex+.50,ey+1.28)],
    fc=C[4],ec=INK,lw=.7,zorder=5))
text(ex+.67,ey+1.51,'Feasible',11.5,ha='center')
route([(ex+.68,ey+1.27),(ex+.68,ey+.98)],lw=.8)
sheet(ex+.30,ey+.43,.68,.51,C[6],.055)
text(ex+.64,ey+.69,'Action',12,True,ha='center')
cost=core['cost_output_port']
route([cost,(16.15,cost[1]),(16.15,ey+2.32),(ex+.24,ey+2.32)])
cp=inputs['constraints_port']
route([cp,(4.38,cp[1]),(4.38,.76),(ex+.69,.76),(ex+.69,ey+.04)],dashed=True,lw=.9)
text(10.0,.48,'Service constraints',11.8,ha='center',color=MUTED)

# The three outcome boxes are siblings driven by the executed action.
OX,OW=18.25,3.40
boxes=[(8.08,2.92),(4.57,3.17),(1.10,3.13)]
for y,h in boxes:card(OX,y,OW,h)
branch_x=18.01
route([(ex+ew,ey+1.82),(branch_x,ey+1.82)],arrow=False)
for y,h in boxes:
    cy=y+h/2
    route([(branch_x,ey+1.82),(branch_x,cy),(OX,cy)])

SERIES=[('worldmoe_final','WorldMoE',C[7],'-',2.05,None),
 ('previous_worldmoe_point','Previous WM',C[0],'--',1.45,'o'),
 ('lgb_recent_point','LightGBM Recent',C[5],'-.',1.45,'s'),
 ('rasc_point','RASC-ME',C[6],':',1.60,'^'),
 ('immediate','Immediate',C[2],'-',1.55,'D')]

text(OX+OW/2,10.67,'Annual cost',15.2,True,ha='center')
text(OX+OW/2,10.34,'2025 · 24 h · US$k',11.5,ha='center',color=MUTED)
bill=subaxis(OX+.14,8.86,OW-.36,1.23)
for j,(key,label,col,*_) in enumerate(SERIES):
    v=MODELS.loc[key,'cost_usd']/1000;yy=4-j
    bill.barh(yy,v,height=.68,color=col,edgecolor='none')
    bill.text(10,yy,label,va='center',fontsize=10.7,fontweight='bold' if j==0 else 'normal',color=INK)
    bill.text(v+8,yy,f'{v:.1f}',va='center',fontsize=11.2,fontweight='bold',color=INK)
bill.set(xlim=(0,650),ylim=(-.6,4.6),yticks=[],xticks=[0,200,400,600])
bill.spines[['top','left','right']].set_visible(False);bill.spines['bottom'].set_color(C[2])
bill.tick_params(axis='x',length=2,labelsize=10)
text(OX+OW/2,8.56,'Saving 18.96% · US$105.3k',12.2,True,ha='center')
text(OX+OW/2,8.29,'79.39% of certified ceiling',11.4,ha='center',color=MUTED)

text(OX+OW/2,7.40,'Diurnal imports',15.2,True,ha='center')
profile=subaxis(OX+.46,5.27,OW-.65,1.36)
hh=CURVES.hour.to_numpy()+.5
for j,(key,label,col,ls,lw,marker) in enumerate(SERIES):
    profile.plot(hh,CURVES[key+'_mw'],label=label,color=col,ls=ls,lw=lw,marker=marker,
                  markevery=(j,5),ms=2.7,mec=MUTED,mew=.25,zorder=6 if j==0 else 3)
profile.set(xlim=(0,24),ylim=(0,1.75),xticks=[0,6,12,18,24],yticks=[0,.5,1,1.5])
profile.spines[['top','right']].set_visible(False)
profile.spines[['bottom','left']].set_color(C[2])
profile.grid(axis='y',color=GRID,lw=.6);profile.set_axisbelow(True)
profile.tick_params(labelsize=10,length=2,pad=2)
profile.set_ylabel('MW',fontsize=11,labelpad=2)
profile.legend(loc='lower center',bbox_to_anchor=(.48,1.02),ncol=2,fontsize=9.7,
               frameon=False,handlelength=1.55,columnspacing=.65,labelspacing=.2)
text(OX+OW/2,4.92,'Local hour (New York)',11,ha='center',color=MUTED)
text(OX+OW/2,4.70,'Full-year means',10.5,ha='center',color=MUTED)

text(OX+OW/2,3.89,'Service windows',15.2,True,ha='center')
text(OX+.21,3.50,'Window',11.3,True)
text(OX+1.78,3.50,'US$k',11.3,True,ha='center')
text(OX+2.89,3.50,'Capture',11.3,True,ha='center')
for row,yy in zip(WINDOWS.sort_values('window_h').itertuples(),[3.04,2.48,1.92]):
    bx=OX+.19;bw=1.06*row.window_h/24
    sheet(bx,yy-.13,bw,.27,C[1],.03)
    text(bx+bw/2,yy,f'{int(row.window_h)} h',11.5,True,ha='center')
    text(OX+1.78,yy,f'{row.saving_usd_mean/1000:.1f}',12.8,True,ha='center')
    text(OX+2.89,yy,f'{row.capture_pct_mean:.2f}%',11.7,ha='center')
text(OX+OW/2,1.49,r'Late $<10^{-8}$ IT MWh',11.2,ha='center')
text(OX+OW/2,1.25,'Closing work = 0',11.2,ha='center')

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=ROOT/'figures/main')
args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
stem='Fig1_mechanism_20261010_v3'
for ext in ['png','svg','pdf']:
    fig.savefig(args.output_dir/f'{stem}.{ext}',dpi=450 if ext=='png' else None,facecolor='white')
fig.savefig(args.output_dir/f'{stem}_preview.png',dpi=130,facecolor='white')
for ext in ['png','svg','pdf']:
    fig.savefig(args.output_dir/f'WorldMoE_architecture_detail_20261010_v3.{ext}',
        dpi=450 if ext=='png' else None,facecolor='white',bbox_inches=Bbox.from_bounds(4.82,1.10,11.18,9.94))
plt.close(fig)
print(args.output_dir/f'{stem}.png')
