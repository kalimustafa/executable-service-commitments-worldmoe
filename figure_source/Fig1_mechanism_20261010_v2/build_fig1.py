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
WINDOWS = pd.read_csv(HERE / 'service_windows.csv')
COMPARATORS = pd.read_csv(HERE / 'comparison_models.csv')
CURVES = pd.read_csv(HERE / 'comparison_diurnal_imports.csv')

INK = '#30303B'
MUTED = '#686571'
LINE = '#D5D3DE'
PURPLE = '#E9687A'
PURPLE_PALE = '#CFCCE3'
TEAL = '#F1837A'
TEAL_PALE = '#F6DFD6'
GOLD = '#F1837A'
GOLD_PALE = '#F6DFD6'
WHITE = '#FFFFFF'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'text.color': INK,
    'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.linewidth': .8, 'savefig.facecolor': WHITE, 'hatch.linewidth': .6})

W, H = 16.2, 9.2
X_OFFSET = 0.0
fig = plt.figure(figsize=(W, H), facecolor=WHITE)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(0, H)); ax.axis('off')

def txt(x, y, text, size=13, weight='normal', color=INK, ha='left', va='center', **kw):
    kw.setdefault('zorder', 12)
    x += X_OFFSET
    return ax.text(x, y, text, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, **kw)

def box(x, y, w, h, fill=WHITE, edge=LINE, radius=.08, lw=.8, z=1):
    x += X_OFFSET
    p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',facecolor=fill,edgecolor=edge,linewidth=lw,zorder=z)
    ax.add_patch(p); return p

def arrow(a,b,color=INK,lw=1.6,style='-|>',curve=0,scale=13,linestyle='-'):
    a=(a[0]+X_OFFSET,a[1]); b=(b[0]+X_OFFSET,b[1])
    p=FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=scale,linewidth=lw,color=color,
        connectionstyle=f'arc3,rad={curve}',linestyle=linestyle,zorder=6)
    ax.add_patch(p); return p

def line(points, color=INK, lw=1, **kw):
    points=[(x+X_OFFSET,y) for x,y in points]
    x,y=zip(*points); return ax.plot(x,y,color=color,lw=lw,**kw)

def inset(x,y,w,h):
    return fig.add_axes([(x+X_OFFSET)/W,y/H,w/W,h/H])

def panel(x,w,label,title):
    box(x,.25,w,8.65,fill='#FFFFFF',edge=LINE,radius=.12,lw=.9)
    box(x,8.32,w,.58,fill='#F3F2F7',edge='none',radius=.12)
    txt(x+.18,8.60,label,17,'bold',PURPLE)
    txt(x+.51,8.60,title,16.2,'bold')

panel(.12,4.42,'a','Data-centre portfolio')
panel(4.83,6.23,'b','WorldMoE architecture')
X_OFFSET=1.8
panel(9.55,4.73,'c','Value and grid response')
X_OFFSET=0.0
arrow((4.56,8.60),(4.81,8.60),scale=15)
arrow((11.09,8.60),(11.33,8.60),scale=15)

# (a) A vector facility, with imports and onsite solar but no invented export link.
txt(.38,8.02,'Per-site physical model',14.3,'bold')

def building(x,y,w,h,depth=.38,dy=.23,shade=PURPLE_PALE):
    ax.add_patch(Polygon([(x-.12,y-.10),(x+w+.09,y-.10),(x+w+depth+.18,y+dy-.08),(x+depth-.09,y+dy-.08)],closed=True,fc='#F1EFF2',ec='none',zorder=2))
    ax.add_patch(Rectangle((x,y),w,h,fc=shade,ec=INK,lw=1,zorder=4))
    ax.add_patch(Polygon([(x+w,y),(x+w+depth,y+dy),(x+w+depth,y+h+dy),(x+w,y+h)],closed=True,fc='#CFCCE3',ec=INK,lw=1,zorder=4))
    ax.add_patch(Polygon([(x,y+h),(x+depth,y+h+dy),(x+w+depth,y+h+dy),(x+w,y+h)],closed=True,fc='#D5D3DE',ec=INK,lw=1,zorder=5))
    for row in range(3):
        for col in range(6):
            ax.add_patch(Rectangle((x+.10+col*(w-.17)/6,y+.18+row*(h-.18)/3),.11,.09,fc='#868397',ec='none',zorder=6))
    # Roof solar modules are mapped to the roof plane, as vector polygons.
    for row in range(2):
        for col in range(5):
            u0=.07+col*.174;u1=u0+.15;v0=.10+row*.43;v1=v0+.33
            pts=[(x+w*u+depth*v,y+h+dy*v) for u,v in [(u0,v0),(u1,v0),(u1,v1),(u0,v1)]]
            ax.add_patch(Polygon(pts,closed=True,fc='#B6B3D6',ec='#FFFFFF',lw=.65,zorder=7))
    ax.add_patch(Rectangle((x+w*.42,y),w*.16,.20,fc='#686571',ec=INK,lw=.6,zorder=7))

building(1.55,6.67,1.84,.86)
building(1.13,6.26,1.25,.56,depth=.29,dy=.18,shade='#D5D3DE')
# Compact transformer and pylon.
ax.add_patch(Rectangle((.39,6.41),.43,.52,fc='#D5D1D1',ec=INK,lw=1,zorder=4))
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
txt(.35,6.10,f"≤ {P['physical']['import_cap_mw']:.2f} MW",13.5,'bold',INK)
# Solar symbol.
ax.add_patch(Circle((3.83,7.82),.16,fc=GOLD,ec='none'))
for theta in np.linspace(0,2*np.pi,8,endpoint=False):
    a=np.array([3.83,7.82])+.23*np.array([np.cos(theta),np.sin(theta)])
    b=np.array([3.83,7.82])+.32*np.array([np.cos(theta),np.sin(theta)])
    line([a,b],GOLD,1.25)
arrow((3.64,7.70),(3.43,7.61),color=GOLD,lw=1.4,scale=12)
txt(3.18,6.10,f"PV {P['physical']['pv_capacity_mw']:.2f} MW",13.5,'bold',TEAL,ha='center')
# Cooling hardware; no claim of an observed efficiency installation.
ax.add_patch(Rectangle((3.62,6.38),.46,.53,fc='#F6DFD6',ec=INK,lw=.8,zorder=7))
for yy in [6.52,6.77]:
    ax.add_patch(Circle((3.85,yy),.087,fc='#F8B2A2',ec=INK,lw=.7,zorder=8))
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
    colour=PURPLE_PALE if name=='NY' else TEAL_PALE if name in ['ME','NH','VT','MA','CT','RI'] else '#F3F2F4'
    for poly in feature['geometry']['coordinates']:
        vertices=[];codes=[]
        for ring in poly:
            a=np.asarray(ring);xx,yy=project(a[:,0],a[:,1]);v=list(zip(xx,yy))
            if len(v)<4:continue
            vertices.extend(v);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(v)-2)+[MPath.CLOSEPOLY])
        if vertices:
            mapax.add_patch(PathPatch(MPath(vertices,codes),fc=colour,ec='#B8B5BE',lw=.68,rasterized=False))
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
        mapax.text(x,y,letter,ha='center',va='center',fontsize=9.8,fontweight='bold',color=INK,zorder=9)
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

# (b) Professional grouped architecture, rendered as native vector geometry.
from professional_architecture import draw_worldmoe_architecture
draw_worldmoe_architecture(ax,4.96,4.03,5.97,4.10)

# Three measured service windows. Duration bands encode allowed service time.
txt(7.945,3.82,r'$q_{t+1}=q_t+a_t-x_t$',17.5,ha='center')
txt(6.49,3.50,'Release → deadline',12.2,'bold',ha='center')
txt(9.04,3.53,'Saving\n(US$k)',11.4,'bold',ha='center',linespacing=.98)
txt(10.25,3.53,'Capture\n(%)',11.4,'bold',ha='center',linespacing=.98)
for hour in [0,12,24]:
    xx=5.18+hour/24*2.60
    line([(xx,2.02),(xx,3.33)],LINE,.8,linestyle=(0,(2,2)))
    txt(xx,1.83,str(hour),10.9,ha='center',color=MUTED)
for row,yy in zip(WINDOWS.sort_values('window_h').itertuples(),[3.04,2.56,2.08]):
    width=row.window_h/24*2.60
    box(5.18,yy,width,.28,fill='#CFCCE3',edge='#B6B3D6',radius=.012,lw=.75)
    line([(5.18+width,yy-.025),(5.18+width,yy+.31)],INK,1.0)
    txt(5.18+width/2,yy+.14,f'{row.window_h:g} h',12.5,'bold',INK,ha='center')
    txt(9.04,yy+.14,f'{row.saving_usd_mean/1000:.1f}',14.1,'bold',INK,ha='center')
    txt(10.25,yy+.14,f'{row.capture_pct_mean:.2f}',13.0,ha='center')
txt(6.48,1.60,'Allowed service time (h)',11.1,color=MUTED,ha='center')
txt(9.59,1.60,'Three-refinement means',10.9,color=MUTED,ha='center')
box(5.18,.70,5.50,.56,fill='#F6DFD6',edge='#D5D1D1',radius=.025,lw=.85)
txt(7.93,1.08,'Execute the current 15 min action',12.8,'bold',ha='center')
txt(7.93,.86,'Capacity · deadlines · power balance',11.3,ha='center')
txt(7.945,.49,r'All windows: late $<10^{-8}$ IT MWh; closing work = 0',12.3,ha='center')
arrow((7.94,1.51),(7.94,1.29),color=INK,lw=1.2,scale=11)
arrow((11.09,1.02),(11.33,1.02),color=INK,scale=13)

X_OFFSET=1.8
# (c) Five archived whole-pipeline comparators, including all available model means.
MODELS = [
    ('worldmoe_final','WorldMoE','#E9687A','-',2.4),
    ('previous_worldmoe_point','Previous WorldMoE','#B6B3D6','--',1.8),
    ('lgb_recent_point','LightGBM Recent','#F8B2A2','-.',1.7),
    ('rasc_point','RASC-ME','#F1837A',':',2.0),
    ('immediate','Immediate','#D5D3DE','-',1.9)
]
txt(9.84,8.02,'2025 · 24 h · archived pipelines',13.2,color=MUTED)
txt(9.84,7.64,f"US${P['saving_usd']:,.0f}",26.2,'bold',PURPLE)
txt(9.84,7.28,f"WorldMoE saving · {P['saving_pct']:.2f}%",13.8)
bill=inset(9.86,5.83,4.09,1.19)
indexed=COMPARATORS.set_index('model_key')
for j,(key,label,colour,style,lw) in enumerate(MODELS):
    value=indexed.loc[key,'cost_usd']/1000;y=4-j
    bill.barh(y,value,height=.70,color=colour,alpha=.88 if j==0 else .78,edgecolor='none')
    bill.text(11,y,label,va='center',fontsize=11.3,color=INK,fontweight='bold' if j==0 else 'normal')
    bill.text(value+9,y,f'{value:.1f}',va='center',fontsize=11.8,fontweight='bold')
bill.set(xlim=(0,625),ylim=(-.65,4.65),yticks=[],xticks=[0,200,400,600])
for s in ['left','top','right']:bill.spines[s].set_visible(False)
bill.spines['bottom'].set_color(LINE);bill.tick_params(axis='x',labelsize=10.5,length=2,pad=3)
txt(13.98,7.12,'Annual cost / US$k',11.0,color=MUTED,ha='right')
txt(9.85,5.36,f"Captured {P['capture_pct']:.2f}% of US$132.6k ceiling",13.2,'bold',PURPLE)

txt(9.85,4.98,'Full-year mean hourly imports',14.0,'bold')
profile=inset(9.98,2.89,3.98,1.25)
h=CURVES.hour.to_numpy()+.5
for j,(key,label,colour,style,lw) in enumerate(MODELS):
    profile.plot(h,CURVES[key+'_mw'],color=colour,lw=lw,ls=style,label=label,
        marker=[None,'o','s','^','D'][j],markevery=(j,5),markersize=3.1,
        markeredgecolor=MUTED,markeredgewidth=.4,
        zorder=5 if key=='worldmoe_final' else 3)
maxval=max(CURVES[key+'_mw'].max() for key, *_ in MODELS)
profile.set(xlim=(0,24),ylim=(0,maxval*1.12),xticks=[0,6,12,18,24])
profile.spines[['top','right']].set_visible(False)
profile.spines[['bottom','left']].set_color('#B7B3BD')
profile.tick_params(labelsize=11,length=2,pad=3)
profile.set_ylabel('MW',fontsize=12,labelpad=1)
profile.grid(axis='y',color='#EFEDEF',lw=.65);profile.set_axisbelow(True)
profile.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,frameon=False,fontsize=10.6,handlelength=1.8,columnspacing=1.0,labelspacing=.35)
txt(11.97,2.56,'Local hour (America/New_York)',11.1,color=MUTED,ha='center')
g=P['grid_worldmoe'];r=P['grid_reference']
txt(9.86,2.18,'WorldMoE vs immediate execution',11.8,'bold',color=MUTED)
txt(9.86,1.87,f"High-price imports −{g['top10_pct']:.2f}%",13.5,'bold',PURPLE)
line([(9.86,1.64),(13.99,1.64)],LINE,.8)
txt(9.86,1.42,'95th-percentile absolute ramp',12.1,'bold')
txt(9.86,1.14,f"{r['p95_abs_ramp_mw']:.4f} → {g['p95_abs_ramp_mw']:.4f} MW / 15 min",12.8,color=PURPLE)
txt(9.86,.80,'Annual hourly peak',12.1,'bold')
txt(9.86,.51,f"{r['portfolio_peak_hourly_mw']:.4f} → {g['portfolio_peak_hourly_mw']:.4f} MW",13.0)

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=PACKAGE/'figures/main')
args=parser.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
stem='Fig1_mechanism_20261010_v2'
for ext in ['png','svg','pdf']:
    fig.savefig(args.output_dir/f'{stem}.{ext}',dpi=500 if ext=='png' else None,facecolor=WHITE,bbox_inches=None)
fig.savefig(args.output_dir/f'{stem}_preview.png',dpi=160,facecolor=WHITE)
from matplotlib.transforms import Bbox
for ext in ['png','svg','pdf']:
    fig.savefig(args.output_dir/f'WorldMoE_architecture_detail_20261010_v2.{ext}',dpi=500 if ext=='png' else None,
        facecolor=WHITE,bbox_inches=Bbox.from_bounds(4.92,4.00,6.05,4.18))
plt.close(fig)
print(str(args.output_dir/f'{stem}.png'))
