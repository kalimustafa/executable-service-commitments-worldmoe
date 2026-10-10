"""Python-only mechanism figure with archived results and Census map geometry.

Run: python figure_source/Fig1_mechanism_20261010/build_fig1.py
The shipped inputs make rendering independent of the large training archives.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
P = json.loads((HERE / 'figure_inputs.json').read_text(encoding='utf-8'))
GEO = json.loads((HERE / 'northeast_states.geojson').read_text(encoding='utf-8'))
PROFILE = pd.read_csv(HERE / 'annual_diurnal_imports.csv')

INK = '#293044'
MUTED = '#627083'
LINE = '#DADDE5'
PURPLE = '#914CA6'
PURPLE_PALE = '#EEE3F2'
TEAL = '#298C87'
TEAL_PALE = '#DCEFEB'
GOLD = '#D9A247'
GOLD_PALE = '#F9EED9'
WHITE = '#FFFFFF'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.linewidth': .8, 'savefig.facecolor': WHITE, 'hatch.linewidth': .6})

W, H = 14.4, 9.2
fig = plt.figure(figsize=(W, H), facecolor=WHITE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H)); ax.axis('off')

def txt(x, y, text, size=13, weight='normal', color=INK, ha='left', va='center', **kw):
    kw.setdefault('zorder', 12)
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kw)

def box(x, y, w, h, fill=WHITE, edge=LINE, radius=.08, lw=.8, z=1):
    p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=fill,edgecolor=edge,linewidth=lw,zorder=z)
    ax.add_patch(p); return p

def arrow(a,b,color=INK,lw=1.6,style='-|>',curve=0,scale=13,linestyle='-'):
    p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=scale,linewidth=lw,color=color,
        connectionstyle=f'arc3,rad={curve}',linestyle=linestyle,zorder=6)
    ax.add_patch(p); return p

def line(points, color=INK, lw=1, **kw):
    x,y=zip(*points); return ax.plot(x,y,color=color,lw=lw,**kw)

def inset(x,y,w,h):
    return fig.add_axes([x/W,y/H,w/W,h/H])

def panel(x,w,label,title):
    box(x,.25,w,8.65,fill='#FCFCFD',edge=LINE,radius=.12,lw=.9)
    box(x,8.32,w,.58,fill='#F0EFF4',edge='none',radius=.12)
    txt(x+.18,8.60,label,17,'bold',PURPLE)
    txt(x+.51,8.60,title,16.2,'bold')

panel(.12,4.42,'a','Data-centre portfolio')
panel(4.83,4.43,'b','Executable-service mechanism')
panel(9.55,4.73,'c','Value and grid response')
arrow((4.56,8.60),(4.81,8.60),scale=15)
arrow((9.29,8.60),(9.53,8.60),scale=15)

# (a) A vector facility, with imports and onsite solar but no invented export link.
txt(.38,8.02,'Per-site physical model',14.3,'bold')

def building(x,y,w,h,depth=.38,dy=.23,shade=PURPLE_PALE):
    ax.add_patch(Polygon([(x-.12,y-.10),(x+w+.09,y-.10),(x+w+depth+.18,y+dy-.08),(x+depth-.09,y+dy-.08)],closed=True,fc='#E3E5EC',ec='none',zorder=2))
    ax.add_patch(Rectangle((x,y),w,h,fc=shade,ec=INK,lw=1,zorder=4))
    ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+dy),(x+w+depth,y+h+dy),(x+w,y+h)],closed=True,fc='#9FA9BD',ec=INK,lw=1,zorder=4))
    ax.add_patch(Polygon([(x,y+h),(x+depth,y+h+dy),(x+w+depth,y+h+dy),(x+w,y+h)],closed=True,fc='#D7DEE8',ec=INK,lw=1,zorder=5))
    for row in range(3):
        for col in range(6):
            ax.add_patch(Rectangle((x+.10+col*(w-.17)/6,y+.18+row*(h-.18)/3),.11,.09,fc='#627189',ec='none',zorder=6))
    # Roof solar modules are mapped to the roof plane, as vector polygons.
    for row in range(2):
        for col in range(5):
            u0=.07+col*.174;u1=u0+.15;v0=.10+row*.43;v1=v0+.33
            pts=[(x+w*u+depth*v,y+h+dy*v) for u,v in [(u0,v0),(u1,v0),(u1,v1),(u0,v1)]]
            ax.add_patch(Polygon(pts,closed=True,fc='#35666F',ec='#E4F1EF',lw=.65,zorder=7))
    ax.add_patch(Rectangle((x+w*.42,y),w*.16,.20,fc='#56647C',ec=INK,lw=.6,zorder=7))

building(1.55,6.67,1.84,.86)
building(1.13,6.26,1.25,.56,depth=.29,dy=.18,shade='#E6E9F0')
# Compact transformer and pylon.
ax.add_patch(Rectangle((.39,6.41),.43,.52,fc='#D8E0E8',ec=INK,lw=1,zorder=4))
for xx in [.45,.58,.71]:
    line([(xx,6.93),(xx,7.05)],INK,.8)
line([(.40,6.36),(.81,6.36)],INK,1.3)
txt(.61,6.68,'↯',25,color=INK,ha='center')
arrow((.82,6.67),(1.11,6.67),color=GOLD,lw=2.4,scale=17)
line([(.42,7.22),(.68,7.73),(.88,7.22)],MUTED,1.05)
for yy,half in [(7.57,.15),(7.43,.23),(7.27,.30)]:
    line([(.66-half,yy),(.66+half,yy)],MUTED,1)
line([(.53,7.23),(.79,7.23)],MUTED,1)
line([(.59,7.48),(.75,7.30),(.57,7.30)],MUTED,.7)
txt(.35,7.95,'',12)
txt(.35,7.02,'Grid import',11.8,'bold')
txt(.35,6.10,f"≤ {P['physical']['import_cap_mw']:.2f} MW",13.5,'bold',GOLD)
# Solar symbol.
ax.add_patch(Circle((3.83,7.82),.16,fc=GOLD,ec='none'))
for theta in np.linspace(0,2*np.pi,8,endpoint=False):
    a=np.array([3.83,7.82])+.23*np.array([np.cos(theta),np.sin(theta)])
    b=np.array([3.83,7.82])+.32*np.array([np.cos(theta),np.sin(theta)])
    line([a,b],GOLD,1.25)
arrow((3.64,7.70),(3.43,7.61),color=GOLD,lw=1.4,scale=12)
txt(3.18,6.10,f"PV {P['physical']['pv_capacity_mw']:.2f} MW",13.5,'bold',TEAL,ha='center')
# Cooling hardware; no claim of an observed efficiency installation.
ax.add_patch(Rectangle((3.62,6.38),.46,.53,fc='#DDE9EA',ec=INK,lw=.8,zorder=7))
for yy in [6.52,6.77]:
    ax.add_patch(Circle((3.85,yy),.087,fc='#A7BAC1',ec=INK,lw=.7,zorder=8))
    for ang in [0,np.pi/2,np.pi,3*np.pi/2]:
        line([(3.85,yy),(3.85+.065*np.cos(ang),yy+.065*np.sin(ang))],INK,.6,zorder=9)
txt(2.30,5.86,f"IT {P['physical']['variable_it_capacity_mw']:.2f} MW  +  idle {P['physical']['idle_it_mw']:.2f} MW",13.6,ha='center')
txt(2.30,5.56,f"PUE {P['physical']['pue_min']:.2f}–{P['physical']['pue_max']:.2f}  ·  zero exports",13,ha='center',color=MUTED)

txt(.38,5.09,'Configured NYISO locations',14.3,'bold')

# Albers equal-area projection for geographic geometry and the four real
# configuration coordinates. This does not assign observed physical facilities.
def project(lon,lat):
    lon,lat=np.deg2rad(lon),np.deg2rad(lat)
    p1,p2,p0,l0=np.deg2rad([41,45,40,-74])
    n=.5*(np.sin(p1)+np.sin(p2)); c=np.cos(p1)**2+2*n*np.sin(p1)
    rho=6371*np.sqrt(c-2*n*np.sin(lat))/n
    rho0=6371*np.sqrt(c-2*n*np.sin(p0))/n
    theta=n*(lon-l0)
    return rho*np.sin(theta),rho0-rho*np.cos(theta)

mapax=inset(.30,1.15,4.04,3.63)
mapax.set_aspect('equal');mapax.axis('off')
for feature in GEO['features']:
    name=feature['properties']['STUSAB']
    colour=PURPLE_PALE if name=='NY' else TEAL_PALE if name in ['ME','NH','VT','MA','CT','RI'] else '#F1F2F4'
    for poly in feature['geometry']['coordinates']:
        vertices=[];codes=[]
        for ring in poly:
            a=np.asarray(ring);xx,yy=project(a[:,0],a[:,1]);v=list(zip(xx,yy))
            if len(v)<4:continue
            vertices.extend(v);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(v)-2)+[MPath.CLOSEPOLY])
        if vertices:
            mapax.add_patch(PathPatch(MPath(vertices,codes),fc=colour,ec='#9BA5B4',lw=.68,rasterized=False))
xx,yy=project(np.array([-81.0,-66.4]),np.array([40.0,47.6]))
mapax.set_xlim(xx[0],xx[1]);mapax.set_ylim(yy[0],yy[1])
for state,lon,lat in [('NY',-76.2,43.3),('VT',-72.75,44.35),('NH',-71.6,43.7),('ME',-69.2,45.35),('MA',-71.2,42.25),('CT',-72.65,41.52),('RI',-71.45,41.50)]:
    x,y=project(lon,lat);mapax.text(x,y,state,ha='center',va='center',fontsize=11.5,color=MUTED)
for letter,site in zip('ABCD',P['sites'].values()):
    x,y=project(site['longitude'],site['latitude'])
    mapax.scatter([x],[y],s=160,color=PURPLE,edgecolors=WHITE,linewidths=.7,zorder=8)
    # Coastal positions are close; label C and D outside their truthful markers.
    if letter in 'CD':
        dx=-60 if letter=='C' else 70;dy=-75 if letter=='C' else -32
        mapax.annotate(letter,(x,y),xytext=(x+dx,y+dy),ha='center',va='center',fontsize=12.5,fontweight='bold',color=PURPLE,
            arrowprops={'arrowstyle':'-','lw':.8,'color':PURPLE},zorder=10)
    else:
        mapax.text(x,y,letter,ha='center',va='center',fontsize=9.8,fontweight='bold',color=WHITE,zorder=9)
x,y=project(-76.1,41.68);mapax.text(x,y,'NYISO',fontsize=13.5,color=PURPLE,fontweight='bold',ha='center')
x,y=project(-68.3,43.75);mapax.text(x,y,'ISO-NE',fontsize=13.5,color=TEAL,fontweight='bold',ha='center')
a=project(-75.3,44.25);b=project(-69.7,44.25)
mapax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,connectionstyle='arc3,rad=-.30',color=TEAL,lw=1.3,linestyle=(0,(3,2))))
x,y=project(-73.5,46.10);mapax.text(x,y,'Frozen-model\ntransfer',fontsize=11.7,color=TEAL,ha='center',va='center')
txt(.40,1.17,'A  WEST · Buffalo',11.9,'bold')
txt(2.38,1.17,'B  CAPITL · Albany',11.9,'bold')
txt(.40,.86,'C  N.Y.C. · New York',11.9,'bold')
txt(2.38,.86,'D  LONGIL · Long Island',11.5,'bold')
txt(.40,.49,'Modelled sites; Census state outlines',11.8,color=MUTED)

# (b) Mechanism: state-conditioned valuation, conserved obligations, execution.
box(5.13,7.48,3.83,.57,fill='#E8EEF5',edge='#C4CFDE')
txt(7.045,7.77,'Issued prices + lagged PV / PUE',14.1,'bold',ha='center')
arrow((7.045,7.45),(7.045,7.19),color=MUTED)
box(5.13,6.29,3.83,.87,fill=PURPLE_PALE,edge='#CBAED5',lw=1)
txt(7.045,6.86,'WorldMoE slot valuation',18.0,'bold',PURPLE,ha='center')
txt(7.045,6.49,'Frozen features + current queue state',12.9,ha='center')
arrow((7.045,6.25),(7.045,6.02),color=PURPLE,lw=1.9)
box(5.13,4.35,3.83,1.61,fill=WHITE,edge='#BFC5D2',lw=1)
txt(7.045,5.67,'Conserved workload ledger',15.3,'bold',ha='center')
txt(7.045,5.22,r'$q_{t+1}=q_t+a_t-x_t$',23,ha='center')
for x,w,text,colour in [(5.30,1.05,'Unfinished',PURPLE_PALE),(6.48,1.05,'Released',GOLD_PALE),(7.67,1.05,'Executed',TEAL_PALE)]:
    box(x,4.62,w,.36,fill=colour,edge='none',radius=.045)
    txt(x+w/2,4.80,text,11.6,ha='center')
arrow((7.045,4.31),(7.045,4.09),color=MUTED)

txt(7.045,3.96,'24 h service window',15.2,'bold',ha='center')
line([(5.35,3.26),(8.77,3.26)],'#B4BDCB',1)
line([(5.35,3.12),(5.35,3.78)],GOLD,1.4)
line([(8.77,3.12),(8.77,3.78)],PURPLE,1.4)
ax.add_patch(Rectangle((5.50,3.29),.61,.34,fc='none',ec=GOLD,lw=1.3,linestyle=(0,(3,2))))
ax.add_patch(Rectangle((7.15,3.29),.61,.34,fc=TEAL,ec=TEAL,lw=1))
arrow((5.98,3.69),(7.45,3.69),color=TEAL,curve=-.16,lw=1.4,scale=12)
txt(5.35,2.98,'Release',12.1,color=MUTED,ha='center')
txt(8.76,2.98,'Deadline',12.1,color=MUTED,ha='center')
txt(7.045,2.62,'Service time moves; work is conserved',12.8,ha='center')
arrow((7.045,2.44),(7.045,2.21),color=MUTED)
box(5.13,1.25,3.83,.95,fill=TEAL_PALE,edge='#A5C8C6',lw=1)
txt(7.045,1.91,'Execute one 15 min action',16.0,'bold',TEAL,ha='center')
txt(7.045,1.56,'Capacity · deadlines · power balance',12.5,ha='center')
# State feedback is a data flow, with remaining work explicitly carried onward.
path=MPath([(5.13,1.73),(4.98,1.73),(4.98,6.72),(5.13,6.72)],[MPath.MOVETO,MPath.LINETO,MPath.LINETO,MPath.LINETO])
ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=10,lw=1.15,color=PURPLE))
txt(7.045,.91,r'Late work $<10^{-8}$ IT MWh',14.0,'bold',ha='center')
txt(7.045,.56,'Closing work: 0',14.0,'bold',ha='center')
arrow((9.28,1.75),(9.53,1.75),color=TEAL,scale=15)

# (c) Every number and curve below is read from the included real result tables.
txt(9.84,8.02,'2025 · 24 h · three refinements',13.4,color=MUTED)
txt(9.84,7.59,f"US${P['saving_usd']:,.0f}",27.5,'bold',PURPLE)
txt(9.84,7.21,f"Annual saving · {P['saving_pct']:.2f}% of reference bill",13.8)
bill=inset(9.86,6.10,4.09,.80)
bill.barh([1,0],[P['reference_cost_usd']/1000,P['policy_cost_usd']/1000],height=.37,color=[GOLD,PURPLE])
bill.set(xlim=(0,620),ylim=(-.46,1.60),yticks=[],xticks=[0,200,400,600])
for s in ['left','top','right']:bill.spines[s].set_visible(False)
bill.spines['bottom'].set_color(LINE);bill.tick_params(axis='x',labelsize=11.2,length=2)
bill.text(0,1.35,'Immediate execution',fontsize=12.0,color=INK)
bill.text(620,1.35,'US$k',ha='right',fontsize=11.5,color=MUTED)
bill.text(0,.35,'WorldMoE',fontsize=12.0,color=INK)
for y,value in zip([1,0],[P['reference_cost_usd']/1000,P['policy_cost_usd']/1000]):
    bill.text(value+9,y,f'{value:.1f}',va='center',fontsize=12.5,fontweight='bold')
box(9.85,5.40,4.10,.14,fill='#E9E3ED',edge='none',radius=.035)
box(9.85,5.40,4.10*P['capture_pct']/100,.14,fill=PURPLE,edge='none',radius=.035)
txt(9.85,5.67,f"{P['capture_pct']:.2f}% of certified opportunity",14.0,'bold',PURPLE)
txt(9.85,5.12,f"Savings ceiling: US${P['savings_ceiling_usd']:,.0f}",12.7,color=MUTED)

txt(9.85,4.72,'Full-year mean hourly imports',14.0,'bold')
profile=inset(9.97,3.03,3.99,1.20)
h=PROFILE.hour.to_numpy()+.5
profile.plot(h,PROFILE.reference_mw,color=GOLD,lw=2,label='Immediate')
profile.plot(h,PROFILE.worldmoe_mw,color=PURPLE,lw=2,label='WorldMoE')
profile.fill_between(h,PROFILE.seed_min_mw,PROFILE.seed_max_mw,color=PURPLE,alpha=.15,lw=0)
profile.set(xlim=(0,24),ylim=(0,max(PROFILE.reference_mw.max(),PROFILE.worldmoe_mw.max())*1.13),xticks=[0,6,12,18,24])
profile.spines[['top','right']].set_visible(False)
profile.spines[['bottom','left']].set_color('#AAB3BF')
profile.tick_params(labelsize=11.3,length=2,pad=3)
profile.set_ylabel('MW',fontsize=12,labelpad=1)
profile.grid(axis='y',color='#E7E9EE',lw=.65)
profile.set_axisbelow(True)
profile.legend(loc='lower center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=11.5,handlelength=1.7,columnspacing=1.4)
txt(11.97,2.71,'Local hour (America/New_York)',11.6,color=MUTED,ha='center')

txt(9.86,2.32,'Highest-price 10% of hours',12.6,'bold')
g=P['grid_worldmoe'];r=P['grid_reference']
txt(9.86,2.02,f"Imports −{g['top10_pct']:.2f}%  ({g['top10_net_mwh']:.1f} MWh)",14.2,'bold',PURPLE)
line([(9.86,1.81),(13.99,1.81)],LINE,.8)
txt(9.86,1.54,'95th-percentile absolute ramp',12.5,'bold')
txt(9.86,1.24,f"{r['p95_abs_ramp_mw']:.4f} → {g['p95_abs_ramp_mw']:.4f} MW / 15 min",13.4,color=PURPLE)
txt(9.86,.85,'Annual hourly peak',12.5,'bold')
txt(9.86,.55,f"{r['portfolio_peak_hourly_mw']:.4f} → {g['portfolio_peak_hourly_mw']:.4f} MW",13.5)

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=PACKAGE/'figures/main')
args=parser.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
stem='Fig1_mechanism_20261010'
for ext in ['png','svg','pdf']:
    fig.savefig(args.output_dir/f'{stem}.{ext}',dpi=500 if ext=='png' else None,facecolor=WHITE,bbox_inches=None)
fig.savefig(args.output_dir/f'{stem}_preview.png',dpi=160,facecolor=WHITE)
plt.close(fig)
print(str(args.output_dir/f'{stem}.png'))
