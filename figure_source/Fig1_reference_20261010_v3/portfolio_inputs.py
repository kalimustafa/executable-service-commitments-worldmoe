"""Parallel portfolio, observation and service-condition inputs for WorldMoE.

The sites are configuration coordinates. A/B/C/D denote WEST/CAPITL/N.Y.C./
LONGIL, respectively; no observed facilities or transmission links are implied.
All glyphs are structural diagrams, never synthetic numerical time series.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath

INK='#30303B'
EDGE='#62616A'
COLORS=['#B6B3D6','#CFCCE3','#D5D3DE','#D5D1D1',
        '#F6DFD6','#F8B2A2','#F1837A','#E9687A']


def draw_portfolio_inputs(fig,ax,x,y,w,h,inputs,geo):
    """Render one input group, nominal size 3.80 x 9.45 inch-like units.

    Return separate observations_port, queue_port and constraints_port.
    Portfolio, market, weather, workload and service inputs are parallel;
    no serial dependency between these inputs is drawn.
    """
    sx,sy=w/3.80,h/9.45
    def p(u,v): return x+u*sx,y+v*sy
    def txt(u,v,s,size=12,weight='normal',ha='left',color=INK):
        return ax.text(*p(u,v),s,fontsize=size,fontweight=weight,
                       ha=ha,va='center',color=color,zorder=15)
    def line(points,color=INK,lw=.8,z=5):
        pts=np.array([p(*v) for v in points]);ax.plot(pts[:,0],pts[:,1],
            color=color,lw=lw,zorder=z)
    def rect(u,v,ww,hh,fill='white',edge=INK,lw=.8,z=4):
        ax.add_patch(Rectangle(p(u,v),ww*sx,hh*sy,facecolor=fill,
                               edgecolor=edge,linewidth=lw,zorder=z))
    def arrow(a,b,color=INK,lw=1):
        ax.add_patch(FancyArrowPatch(p(*a),p(*b),arrowstyle='-|>',
                      mutation_scale=9,color=color,lw=lw,zorder=10))
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle=f'round,pad=0,rounding_size={.04*min(sx,sy)}',
        facecolor='white',edgecolor=EDGE,lw=1.0,zorder=1))
    txt(1.90,9.10,'Portfolio & inputs',15.0,'bold',ha='center')

    # Flat facility section, placed beside the geographical panel.
    rect(.63,7.30,1.14,.75)
    line([(.57,8.05),(1.82,8.05)],lw=1.2)
    line([(.17,7.29),(1.85,7.29)],lw=1.0)
    for j in range(4):
        u=.66+j*.265
        vertices=[p(u,8.10),p(u+.20,8.10),p(u+.25,8.27),p(u+.05,8.27)]
        ax.add_patch(Polygon(vertices,facecolor=COLORS[0],edgecolor=INK,lw=.6,zorder=5))
    rect(.20,7.40,.24,.40,COLORS[3],lw=.7)
    for u in [.24,.32,.40]:line([(u,7.8),(u,7.90)],lw=.65)
    arrow((.44,7.62),(.68,7.62),COLORS[7],1.15)
    line([(.69,7.62),(.69,7.93),(1.66,7.93)],COLORS[7],1.1)
    arrow((1.24,8.05),(1.24,7.93),COLORS[7],1.0)
    for u in [.77,1.10,1.43]:
        rect(u,7.35,.23,.42,COLORS[1],lw=.65)
        for j in range(4):
            line([(u+.03,7.40+j*.09),(u+.20,7.40+j*.09)],'white',1.2)
        arrow((u+.115,7.93),(u+.115,7.78),COLORS[7],.8)
    txt(1.00,8.56,'Per-site model',11.8,'bold',ha='center')

    def project(lon,lat):
        lon,lat=np.deg2rad(lon),np.deg2rad(lat)
        p1,p2,p0,l0=np.deg2rad([41,45,40,-74])
        n=.5*(np.sin(p1)+np.sin(p2));c=np.cos(p1)**2+2*n*np.sin(p1)
        rho=6371*np.sqrt(c-2*n*np.sin(lat))/n
        rho0=6371*np.sqrt(c-2*n*np.sin(p0))/n
        theta=n*(lon-l0)
        return rho*np.sin(theta),rho0-rho*np.cos(theta)
    low=ax.transData.transform(p(1.97,7.02));high=ax.transData.transform(p(3.50,8.58))
    fb=fig.transFigure.inverted().transform([low,high])
    mapax=fig.add_axes([fb[0,0],fb[0,1],fb[1,0]-fb[0,0],fb[1,1]-fb[0,1]])
    mapax.set_aspect('equal');mapax.axis('off');mapax.patch.set_alpha(0)
    for feature in geo['features']:
        state=feature['properties']['STUSAB']
        if state not in ['NY','VT','NH','ME','MA','CT','RI']:continue
        for poly in feature['geometry']['coordinates']:
            vertices=[];codes=[]
            for ring in poly:
                a=np.asarray(ring);xx,yy=project(a[:,0],a[:,1]);pts=list(zip(xx,yy))
                if len(pts)<4:continue
                vertices.extend(pts);codes.extend([MPath.MOVETO]+[MPath.LINETO]*(len(pts)-2)+[MPath.CLOSEPOLY])
            if vertices:
                mapax.add_patch(PathPatch(MPath(vertices,codes),
                    facecolor=COLORS[1] if state=='NY' else COLORS[4],
                    edgecolor=EDGE,lw=.45))
    xx,yy=project(np.array([-80.60,-66.20]),np.array([39.80,47.70]))
    mapax.set_xlim(xx[0],xx[1]);mapax.set_ylim(yy[0],yy[1])
    offsets={'A':(-.40,1.10),'B':(-.20,1.10),'C':(-1.72,-.48),'D':(1.75,-.48)}
    for letter,site in zip('ABCD',inputs['sites'].values()):
        xp,yp=project(site['longitude'],site['latitude']);dx,dy=offsets[letter]
        xt,yt=project(site['longitude']+dx,site['latitude']+dy)
        mapax.scatter([xp],[yp],marker='s',s=14,color=INK,edgecolors='white',linewidths=.35,zorder=8)
        mapax.annotate(letter,(xp,yp),(xt,yt),ha='center',va='center',fontsize=10.5,
            fontweight='bold',color=INK,arrowprops={'arrowstyle':'-','color':INK,'lw':.5},zorder=10)
    txt(2.73,8.76,'4 modelled sites',11.8,'bold',ha='center')
    physical=inputs['physical']
    txt(1.86,6.72,f"IT {physical['variable_it_capacity_mw']:.2f} MW  ·  PV {physical['pv_capacity_mw']:.2f} MW",12.0,ha='center')
    txt(1.86,6.38,f"Grid import ≤ {physical['import_cap_mw']:.2f} MW per site",12.0,ha='center')
    line([(.16,6.09),(3.50,6.09)],COLORS[2],.8)

    # Three parallel bands. Stacked record sheets identify tensors without
    # introducing a serial data-processing chain or invented sample values.
    bands=[('Market','Issued prices',5.41,COLORS[0]),
           ('Weather','Lagged PV / PUE',4.34,COLORS[5]),
           ('Workload','Released work',3.27,COLORS[2])]
    for label,detail,cy,color in bands:
        rect(.18,cy-.37,3.12,.79,'white',COLORS[2],.65)
        # Only sheet offsets use a restrained pseudo-depth cue.
        for off in [2,1,0]:
            rect(.34+off*.055,cy-.21+off*.047,.64,.41,
                 'white' if off else color,EDGE,.55,z=5+(2-off))
        for row in range(3):
            for col in range(4):
                rect(.385+col*.135,cy-.155+row*.107,.109,.073,
                     'white',EDGE,.32,z=9)
        txt(1.27,cy+.17,label,12.8,'bold')
        txt(1.27,cy-.17,detail,11.5)
        line([(3.30,cy),(3.62,cy)],INK,.8)
    # A common observations bus joins parallel sources to the architecture.
    line([(3.62,3.27),(3.62,7.25),(3.80,7.25)],INK,.85)
    line([(.16,2.62),(3.50,2.62)],COLORS[2],.8)
    txt(.22,2.30,'Service conditions',12.8,'bold')

    # Three release-to-deadline lengths; the bars are declared service windows,
    # not observed job schedules. The ledger is a distinct queue-state input.
    for hh,cy,color in [(12,1.74,COLORS[0]),(16,1.29,COLORS[5]),(24,.84,COLORS[7])]:
        txt(.22,cy,f'{hh} h',11.5)
        rect(.78,cy-.065,1.15*(hh/24),.13,color,EDGE,.45)
        line([(.78,cy-.13),(.78,cy+.13)],INK,.6)
    txt(2.66,1.91,'Queue',11.5,'bold',ha='center')
    rect(2.23,.81,.84,.80,'white',INK,.7)
    for row in range(4):
        for col in range(3):
            rect(2.30+col*.22,.89+row*.165,.17,.12,
                 COLORS[1] if row<2 else COLORS[4],EDGE,.4)
    line([(3.07,1.21),(3.40,1.21),(3.40,1.90),(3.80,1.90)],INK,.85)
    line([(.78,.56),(.78,.45),(3.80,.45)],INK,.85)
    return {'observations_port':p(3.80,7.25),
            'queue_port':p(3.80,1.90),
            'constraints_port':p(3.80,.45),
            'map_axes':mapax,'bounds':(x,y,w,h)}


if __name__=='__main__':
    here=Path(__file__).resolve().parent;source=here
    inputs=json.loads((source/'figure_inputs.json').read_text(encoding='utf-8'))
    geo=json.loads((source/'northeast_states.geojson').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(4.40,10.05),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,4.40),ylim=(0,10.05));ax.axis('off')
    draw_portfolio_inputs(fig,ax,.30,.30,3.80,9.45,inputs,geo)
    for ext in ['png','pdf','svg']:
        fig.savefig(here/f'portfolio_inputs_preview.{ext}',dpi=190,facecolor='white')
    plt.close(fig)
