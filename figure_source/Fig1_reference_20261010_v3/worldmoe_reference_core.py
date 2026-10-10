"""WorldMoE core in five aligned, nested rows inspired by the supplied layout.

Rows are sequential computational stages, not cross-validation folds.  The
right/row gutters carry actual dependencies; APIC prices additionally bypass
queue-conditioned attention to the cost decoder. Highlighted expert routes are
schematic. The final port contains slot-cost evidence, not executable actions.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon,Circle,FancyArrowPatch
from matplotlib.colors import to_rgb,to_hex

PALETTE=['#B6B3D6','#CFCCE3','#D5D3DE','#D5D1D1',
         '#F6DFD6','#F8B2A2','#F1837A','#E9687A']
INK='#3E3C43';WHITE='#FFFFFF'

def tint(colour,strength):
    return to_hex(tuple(1-strength*(1-c) for c in to_rgb(colour)))

def draw_worldmoe_core(ax,x,y,w,h):
    """Draw inside an 8.4 x 9.9 reference box, returning the three flow ports."""
    sx=w/8.4;sy=h/9.9;C=PALETTE;artists=[]
    def xy(a,b):return x+a*sx,y+b*sy
    def txt(a,b,label,fs=12,bold=False,ha='center',rotation=0):
        p=ax.text(*xy(a,b),label,ha=ha,va='center',fontsize=fs,
                  fontweight='bold' if bold else 'normal',color=INK,
                  rotation=rotation,linespacing=1.12,zorder=30)
        artists.append(p);return p
    def rect(a,b,ww,hh,fc=WHITE,ec=INK,lw=.75,z=6):
        p=Rectangle(xy(a,b),ww*sx,hh*sy,fc=fc,ec=ec,lw=lw,zorder=z)
        ax.add_patch(p);artists.append(p);return p
    def poly(points,fc=WHITE,lw=.75,z=10,fold=False,fold_colour=None):
        if fold:
            base=C[1] if fold_colour is None else fold_colour
            # Two narrow visible faces, with the same right/up offset as all
            # other folded sheets. The original front face stays at its port.
            dx,dy=.045,.040
            for j,strength in [(0,.34),(1,.59)]:
                p,q=points[j],points[(j+1)%len(points)]
                side=[p,q,(q[0]+dx,q[1]+dy),(p[0]+dx,p[1]+dy)]
                f=Polygon([xy(*v) for v in side],closed=True,fc=tint(base,strength),
                          ec=INK,lw=.45,zorder=z-1)
                ax.add_patch(f);artists.append(f)
        p=Polygon([xy(*q) for q in points],closed=True,fc=fc,ec=INK,lw=lw,zorder=z)
        ax.add_patch(p);artists.append(p);return p
    def folded_rect(a,b,ww,hh,fc=WHITE,base=None,depth=.045):
        base=C[1] if base is None else base;dy=depth*.89
        poly([(a,b+hh),(a+ww,b+hh),(a+ww+depth,b+hh+dy),(a+depth,b+hh+dy)],
             tint(base,.34),lw=.5,z=4)
        poly([(a+ww,b),(a+ww+depth,b+dy),(a+ww+depth,b+hh+dy),(a+ww,b+hh)],
             tint(base,.61),lw=.5,z=4)
        return rect(a,b,ww,hh,fc,lw=.8)
    def circle(a,b,r=.075,label='',fs=12,fc=WHITE):
        p=Circle(xy(a,b),r*min(sx,sy),fc=fc,ec=INK,lw=.75,zorder=24)
        ax.add_patch(p);artists.append(p)
        if label:txt(a,b,label,fs)
    def line(points,arrow=True,dash=False,lw=.9):
        if len(points)>2:
            p,=ax.plot(*zip(*(xy(*q) for q in points[:-1])),color=INK,lw=lw,
                       ls=(0,(2,1.6)) if dash else '-',zorder=18)
            artists.append(p)
        p=FancyArrowPatch(xy(*points[-2]),xy(*points[-1]),arrowstyle='-|>' if arrow else '-',
                          mutation_scale=9.5,color=INK,lw=lw,shrinkA=0,shrinkB=0,
                          linestyle=(0,(2,1.6)) if dash else '-',zorder=18)
        ax.add_patch(p);artists.append(p)
    def box(a,b,ww,hh,label,fs=12,fc=WHITE,bold=False):
        rect(a,b,ww,hh,fc);txt(a+ww/2,b+hh/2,label,fs,bold)
    def matrix(a,b,nc,nr,dx=.115,dy=.085,col=None,stack=False):
        col=C[1] if col is None else col
        if stack:
            ww=(nc-.27)*dx+.042;hh=(nr-.32)*dy+.042
            for depth in (2,1):
                rect(a-.021+.032*depth,b-.021+.028*depth,ww,hh,
                     tint(col,.18),tint(C[0],.64),.38,z=11)
            rect(a-.021,b-.021,ww,hh,WHITE,tint(C[0],.75),.42,z=12)
        for rr in range(nr):
            for cc in range(nc):rect(a+cc*dx,b+rr*dy,dx*.73,dy*.68,col,lw=.3,z=16)
    def heads(a,b,ww=.60,hh=.10,col=None):
        for k in range(4):rect(a+k*ww/4,b,ww*.18,hh,C[0] if col is None else col,lw=.35,z=16)

    # One parent with five identical row frames, aligned input/module/output.
    rect(.025,.035,8.35,9.83,tint(C[1],.12),tint(C[0],.84),1.0,z=1)
    txt(4.20,9.47,'WorldMoE architecture',16.5,True)
    row_y=[7.41,5.63,3.85,2.07,.29]
    labels=['Encoding','World layers','APIC','Attention','Cost']
    for b,label in zip(row_y,labels):
        rect(.23,b,7.82,1.60,tint(C[1],.27),ec='none',z=2)
        txt(.40,b+.80,label,14.0,True,rotation=90)
        folded_rect(.62,b+.24,1.18,1.12,tint(C[3],.43),C[3],.035)
        folded_rect(2.04,b+.16,4.60,1.28,WHITE,C[1],.050)
        folded_rect(6.90,b+.24,1.11,1.12,tint(C[4],.66),C[4],.035)
        line([(1.80,b+.80),(2.04,b+.80)])
        line([(6.64,b+.80),(6.90,b+.80)])

    # Sequential dependencies use row gutters and enter the next input card
    # at its top. They never pass through another stage's nested module.
    for i in range(4):
        b=row_y[i];n=row_y[i+1]
        line([(8.01,b+.80),(8.175,b+.80),(8.175,b-.085),
              (1.21,b-.085),(1.21,n+1.36)],lw=1.0)

    # 1. Matrix tokens, projection, weighted multi-scale fusion.
    b=row_y[0]
    txt(1.21,b+1.17,'3 scales',13.2,True)
    for depth in (2,1,0):matrix(.79+.043*depth,b+.43+.090*depth,6,3,.12,.073,C[1])
    poly([(2.27,b+1.08),(3.51,b+1.08),(3.27,b+.53),(2.51,b+.53)],
         tint(C[1],.64),fold=True,fold_colour=C[1])
    txt(2.89,b+.805,'8 → 256',13)
    line([(2.04,b+.80),(2.27,b+.80)])
    # Three incoming scales merge; typed embeddings form a separate branch.
    for yy,col in [(b+1.08,C[1]),(b+.80,C[0]),(b+.52,C[2])]:
        rect(3.78,yy-.075,.39,.15,col,lw=.5)
        line([(3.51,b+.80),(3.66,b+.80),(3.66,yy),(3.78,yy)],lw=.7)
        line([(4.17,yy),(4.39,yy),(4.39,b+.80),(4.64,b+.80)],lw=.7)
    circle(4.74,b+.80,.10,'Σ',14)
    matrix(4.97,b+1.15,5,1,.11,.075,C[2])
    line([(5.24,b+1.13),(5.24,b+.885)],lw=.7)
    circle(5.24,b+.80,.085,'+',13)
    line([(4.84,b+.80),(5.155,b+.80)])
    matrix(5.69,b+.54,5,5,.125,.10,C[1],stack=True)
    line([(5.325,b+.80),(5.66,b+.80)])
    line([(6.31,b+.80),(6.64,b+.80)])
    txt(7.455,b+1.09,'40 tokens',13.0,True)
    matrix(7.08,b+.51,5,3,.15,.12,C[1],stack=True)
    txt(7.455,b+.37,'×256',12.2)
    data_port=xy(0,b+.09)
    line([(0,b+.09),(1.21,b+.09),(1.21,b+.24)],lw=1.0)
    # Shared observed/issued inputs are reused by the later market, physical,
    # and state paths; a preceding row's output is not their complete input.
    # This single bus stays left of the vertical row labels, then enters each
    # affected input card from below. Queue remains an independent port.
    line([(.105,b+.09),(.105,row_y[4]+.15)],False,lw=.72)
    for downstream in (row_y[2],row_y[3],row_y[4]):
        line([(.105,downstream+.15),(1.21,downstream+.15),
              (1.21,downstream+.24)],lw=.72)

    # 2. Ten world layers; the sparse sublayer is nested below the depth bar.
    b=row_y[1]
    txt(1.21,b+1.15,'Tokens',13.2,True)
    matrix(.84,b+.52,5,4,.145,.105,C[1],stack=True)
    for k in range(10):
        rect(2.22+k*.412,b+1.03,.33,.23,C[5] if k%2 else tint(C[1],.63),lw=.5)
        txt(2.385+k*.412,b+1.145,str(k+1),11.2)
    line([(2.04,b+.80),(2.13,b+.80),(2.13,b+1.145),(2.22,b+1.145)],lw=.75)
    line([(6.258,b+1.145),(6.50,b+1.145),(6.50,b+.80),(6.64,b+.80)],lw=.75)
    # Shared, Top-2 general, and independently gated tail are parallel paths.
    box(2.27,b+.31,.82,.24,'Shared',11.3,tint(C[1],.48))
    txt(3.46,b+.65,'Top-2',11.5,True)
    poly([(3.11,b+.53),(3.80,b+.53),(3.64,b+.26),(3.27,b+.26)],
         tint(C[1],.58),fold=True,fold_colour=C[1])
    for k in range(7):
        rect(4.038+k*.22,b+.385,.16,.19,tint(C[1],.24),lw=.35,z=11)
        rect(4.01+k*.22,b+.36,.16,.19,C[6] if k in (1,4) else C[1],lw=.4,z=16)
    line([(3.80,b+.42),(4.00,b+.42)],lw=.7)
    circle(6.08,b+.49,.088,'Σ',12)
    line([(5.49,b+.455),(5.99,b+.455)],lw=.7)
    line([(3.09,b+.43),(3.09,b+.205),(6.08,b+.205),(6.08,b+.402)],lw=.7)
    circle(6.08,b+.795,.065,'σ',12,fc=tint(C[5],.62))
    rect(6.015,b+.64,.13,.065,C[5],lw=.4,z=16)
    line([(6.08,b+.73),(6.08,b+.705)],dash=True,lw=.7)
    line([(6.08,b+.64),(6.08,b+.578)],dash=True,lw=.7)
    line([(2.795,b+1.03),(2.795,b+.90),(3.46,b+.90),(3.46,b+.735)],dash=True,lw=.6)
    txt(7.455,b+1.105,'Frozen',13.0,True)
    txt(7.455,b+.88,'features',12.8)
    matrix(7.075,b+.43,5,3,.15,.10,C[1],stack=True)

    # 3. APIC: DA and DART jointly determine RT; published DA is an anchor.
    b=row_y[2]
    txt(1.21,b+1.145,'R2',13.5,True)
    txt(1.21,b+.93,'+ market',11.5)
    matrix(.84,b+.36,5,4,.145,.105,C[1],stack=True)
    for yy,label,col in [(b+1.03,'DA',C[1]),(b+.48,'DART',C[5])]:
        poly([(2.32,yy+.16),(3.18,yy+.16),(3.04,yy-.16),(2.46,yy-.16)],
             tint(col,.63),fold=True,fold_colour=col)
        txt(2.75,yy,label,11.8,True)
    line([(2.04,b+.80),(2.20,b+.80),(2.20,b+1.03),(2.32,b+1.03)],lw=.7)
    line([(2.20,b+.80),(2.20,b+.48),(2.32,b+.48)],lw=.7)
    rect(3.78,b+.83,.89,.42,tint(C[1],.30),lw=.6)
    txt(4.225,b+1.12,'PRE',11.5,True)
    matrix(3.95,b+.885,4,1,.14,.08,C[1])
    rect(3.78,b+.28,.89,.42,tint(C[4],.46),lw=.6)
    txt(4.225,b+.58,'POST',11.5,True)
    # Anchor symbol in the post-publication branch.
    circle(4.225,b+.435,.023)
    line([(4.225,b+.41),(4.225,b+.34)],False,lw=.85)
    line([(4.10,b+.385),(4.35,b+.385)],False,lw=.85)
    line([(4.10,b+.35),(4.15,b+.315),(4.225,b+.30),(4.30,b+.315),(4.35,b+.35)],False,lw=.8)
    line([(3.18,b+1.03),(3.78,b+1.03)],lw=.7)
    line([(3.18,b+.48),(3.50,b+.48),(3.50,b+.94),(3.78,b+.94)],lw=.7)
    line([(3.50,b+.48),(3.78,b+.48)],lw=.7)
    circle(5.13,b+.80,.09,'Q',12.5)
    line([(4.67,b+1.03),(5.13,b+1.03),(5.13,b+.89)],lw=.7)
    line([(4.67,b+.48),(5.13,b+.48),(5.13,b+.71)],lw=.7)
    circle(5.86,b+.80,.085,'+',13)
    line([(5.22,b+.80),(5.775,b+.80)])
    box(5.46,b+1.13,.66,.23,'δ',13,tint(C[1],.30))
    line([(2.20,b+.80),(2.20,b+1.38),(5.79,b+1.38),(5.79,b+1.36)],lw=.6)
    line([(5.79,b+1.13),(5.79,b+.96),(5.86,b+.96),(5.86,b+.885)],lw=.7)
    line([(5.945,b+.80),(6.64,b+.80)])
    txt(7.455,b+1.115,'7 RT + h',12.7,True)
    for k in range(7):rect(7.035+k*.12,b+.49,.080,.13+k*.032,C[5],lw=.35)

    # 4. Two Q/K/V directions, two temporal layers and explicit queue queries.
    b=row_y[3]
    txt(1.21,b+1.15,'RT + h',12.5,True)
    txt(1.21,b+.93,'+ physics',11.5)
    matrix(.84,b+.39,5,3,.145,.10,C[1],stack=True)
    for yy,label,col in [(b+1.07,'M',C[1]),(b+.51,'P',C[3])]:
        poly([(2.24,yy+.15),(2.89,yy+.15),(2.78,yy-.15),(2.35,yy-.15)],
             tint(col,.65),fold=True,fold_colour=col)
        txt(2.565,yy,label,12.5,True)
    line([(2.04,b+.80),(2.15,b+.80),(2.15,b+1.07),(2.24,b+1.07)],lw=.65)
    line([(2.15,b+.80),(2.15,b+.51),(2.24,b+.51)],lw=.65)
    for yy in (b+1.07,b+.51):
        rect(3.26,yy-.18,.75,.36,WHITE,lw=.65)
        txt(3.635,yy+.065,'Q / KV',11.0)
        heads(3.355,yy-.115,.59,.055)
    line([(2.89,b+1.07),(3.26,b+1.07)],lw=.65)
    line([(2.89,b+.51),(3.26,b+.51)],lw=.65)
    line([(2.99,b+1.07),(2.99,b+.60),(3.26,b+.60)],lw=.6)
    line([(3.09,b+.51),(3.09,b+.98),(3.26,b+.98)],lw=.6)
    circle(4.27,b+.80,.068,'+',11.5)
    line([(4.01,b+1.07),(4.27,b+1.07),(4.27,b+.868)],lw=.65)
    line([(4.01,b+.51),(4.27,b+.51),(4.27,b+.732)],lw=.65)
    rect(4.64,b+.615,.63,.53,tint(C[1],.28),lw=.65)
    folded_rect(4.58,b+.555,.63,.53,WHITE,C[1],.035)
    heads(4.655,b+.865,.50,.055)
    for k in range(3):rect(4.675+k*.145,b+.65,.095,.10,C[5],lw=.3,z=16)
    txt(4.935,b+1.285,'×2',12.0,True)
    line([(4.338,b+.80),(4.58,b+.80)],lw=.75)
    rect(5.60,b+.58,.76,.44,tint(C[1],.35),lw=.7)
    txt(5.98,b+.91,'Q / KV',11.0)
    heads(5.675,b+.665,.61,.06)
    line([(5.21,b+.80),(5.60,b+.80)],lw=.75)
    line([(6.36,b+.80),(6.64,b+.80)],lw=.75)
    txt(5.98,b+.30,'Queue',11.8,True)
    line([(5.98,b+.40),(5.98,b+.58)],lw=.7)
    queue_port=xy(0,b+.08)
    line([(0,b+.08),(5.98,b+.08),(5.98,b+.205)],lw=.9)
    txt(7.455,b+1.13,'Context',13.0,True)
    matrix(7.08,b+.47,5,4,.15,.11,C[1],stack=True)

    # 5. State pooling, contraction decoder and separate cross-moment branch.
    b=row_y[4]
    txt(1.21,b+1.15,'State',13.0,True)
    txt(1.21,b+.93,'+ context',11.5)
    matrix(.84,b+.39,5,3,.145,.10,C[1],stack=True)
    for yy,label in [(b+1.18,'local'),(b+.80,'mean'),(b+.42,'max')]:
        box(2.24,yy-.13,.71,.26,label,11.3,tint(C[1],.32))
        line([(2.04,b+.80),(2.13,b+.80),(2.13,yy),(2.24,yy)],lw=.65)
        line([(2.95,yy),(3.19,yy),(3.19,b+.80),(3.36,b+.80)],lw=.65)
    for a,hh,label,col in [(3.45,.88,'327',C[1]),(3.98,.70,'96',C[1]),
                           (4.51,.53,'64',C[0]),(5.04,.36,'5',C[5])]:
        poly([(a,b+.80-hh/2),(a,b+.80+hh/2),(a+.34,b+.80+hh*.38),(a+.34,b+.80-hh*.38)],
             tint(col,.67),fold=True,fold_colour=col)
        txt(a+.17,b+.80,label,11.5,True)
    for a,z in [(3.36,3.45),(3.79,3.98),(4.32,4.51),(4.85,5.04)]:
        line([(a,b+.80),(z,b+.80)],lw=.7)
    circle(5.98,b+.80,.095,'Σ',13)
    line([(5.38,b+.80),(5.885,b+.80)],lw=.75)
    # Rho has its own small neural branch and enters physical slot cost.
    for k in range(3):rect(5.33+k*.18,b+.31,.11,.12,C[1],lw=.3,z=16)
    txt(5.89,b+.36,'ρ',13.5,True)
    line([(5.98,b+.43),(5.98,b+.705)],lw=.7)
    line([(6.075,b+.80),(6.64,b+.80)],lw=.75)
    txt(7.455,b+1.115,'Slot costs',12.7,True)
    for offset in (0,.14,.28):
        line([(7.08,b+.49+offset),(7.31,b+.49+offset),(7.46,b+.57+offset),(7.79,b+.57+offset)],False,lw=.75)
    # Prices bypass the entire queue-conditioned attention stage, entering
    # the cost module at its upper edge through a separate right-side gutter.
    a=row_y[2]
    line([(8.01,a+.80),(8.29,a+.80),(8.29,b+1.51),
          (5.98,b+1.51),(5.98,b+.895)],lw=.9)
    txt(5.74,b+1.24,'RT',11.0)
    cost_port=xy(8.01,b+.80)
    return {'artists':artists,'data_input_port':data_port,
            'queue_input_port':queue_port,'cost_output_port':cost_port,
            'row_order':['Encoding','World layers','APIC','Attention','Cost'],
            'shared_input_bus':'observed/issued inputs also feed R2, physics, and state; prior-row outputs are partial inputs',
            'output_semantics':'slot-cost evidence; external deterministic executor required',
            'route_semantics':'illustrative expert routes, not observed gating weights'}


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none',
                         'pdf.fonttype':42,'savefig.facecolor':WHITE})
    fig=plt.figure(figsize=(8.4,9.9),facecolor=WHITE)
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,8.4),ylim=(0,9.9));ax.axis('off')
    draw_worldmoe_core(ax,0,0,8.4,9.9)
    root=Path(__file__).resolve().parent
    for suffix in ('png','pdf','svg'):
        fig.savefig(root/f'worldmoe_reference_core_preview.{suffix}',dpi=220)
    plt.close(fig)
