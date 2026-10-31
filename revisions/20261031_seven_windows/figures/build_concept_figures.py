"""Native vector mechanisms for the revised main manuscript.

The full network diagram is retained in the supplement. These main-text
diagrams expose the economic mechanism without changing model facts or
inventing numerical certificate values.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT=Path(__file__).resolve().parent
C={'purple':'#a34fbe','pink':'#e563a0','lightpink':'#edacd0',
   'lavender':'#e6d5ed','pale':'#f6eff8','text':'#38223f',
   'axis':'#5d5662','grid':'#eeeaf0','white':'#ffffff'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.2,
 'text.color':C['text'],'svg.fonttype':'none','pdf.fonttype':42,
 'mathtext.fontset':'dejavusans','savefig.facecolor':'white'})

def setup(height):
    fig=plt.figure(figsize=(7.2,height));ax=fig.add_axes([0,0,1,1])
    ax.set_xlim(0,7.2);ax.set_ylim(0,height);ax.axis('off')
    return fig,ax

def box(ax,x,y,w,h,fill='white',edge=None,r=.06,lw=.9):
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
        fc=fill,ec=edge or C['axis'],lw=lw)
    ax.add_patch(p);return p

def text(ax,x,y,s,size=9.2,weight='normal',ha='center',va='center',color=None,spacing=1.35):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,ha=ha,va=va,
        color=color or C['text'],linespacing=spacing)

def arrow(ax,xy1,xy2,color=None,lw=1.15,style='-|>',ms=8):
    ax.add_patch(FancyArrowPatch(xy1,xy2,arrowstyle=style,mutation_scale=ms,
        lw=lw,color=color or C['purple'],shrinkA=0,shrinkB=0))

def elbow(ax,points,color=None,lw=1.15):
    color=color or C['purple']
    for a,b in zip(points[:-2],points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,solid_capstyle='round')
    arrow(ax,points[-2],points[-1],color,lw)

def export(fig,stem):
    for suffix in ('svg','pdf','png'):
        fig.savefig(OUT/f'{stem}.{suffix}',dpi=600,facecolor='white')
    plt.close(fig)

fig,ax=setup(4.0)
text(ax,.15,3.78,'WorldMoE from decision-time information to settled service',size=12,
     weight='bold',ha='left')

# Input columns are aligned with their destinations; all output paths occupy
# separate inter-box corridors rather than crossing other information paths.
text(ax,.84,3.40,'Available inputs',size=10,weight='bold')
box(ax,.14,2.49,1.40,.67,C['pale'])
text(ax,.84,2.825,'Market information\nPublication status',size=9.2)
box(ax,.14,1.37,1.40,.88,C['pale'])
text(ax,.84,1.81,'Available PV / PUE\nReleased work\nOutstanding service',size=9.2)

text(ax,3.37,3.40,'Frozen WorldMoE',size=10,weight='bold')
box(ax,1.91,2.45,2.91,.77,C['lavender'],C['purple'])
text(ax,3.365,3.025,'Market representation',size=10,weight='bold')
text(ax,3.365,2.70,'W0/W1 sparse MoE + R2 / APIC\nConsistent price-distribution signals',size=9.2)
box(ax,1.91,1.37,2.91,.80,C['pale'],C['purple'])
text(ax,3.365,1.96,'Physical and queue valuation',size=10,weight='bold')
text(ax,3.365,1.64,'Expected work-execution cost\nFuture capacity and obligation value',size=9.2)
arrow(ax,(1.54,2.825),(1.91,2.825))
arrow(ax,(1.54,1.81),(1.91,1.81))
arrow(ax,(3.365,2.45),(3.365,2.17))

box(ax,5.21,3.19,1.82,.39,C['lightpink'],C['pink'])
text(ax,6.12,3.385,'Authorised window W',size=9.2,weight='bold')
box(ax,5.21,1.37,1.82,1.55,C['pale'],C['pink'])
text(ax,6.12,2.68,'Contract executor',size=10,weight='bold')
text(ax,6.12,2.29,'Release and deadlines\nSite and grid capacity',size=9.2)
text(ax,6.12,1.77,'Commit the current\n15 min service action',size=9.2,weight='bold')
arrow(ax,(6.12,3.19),(6.12,2.92),C['pink'])
elbow(ax,[(4.82,1.81),(5.01,1.81),(5.01,2.25),(5.21,2.25)])

box(ax,.14,.20,3.0,.73,C['white'],C['axis'])
text(ax,1.64,.72,'Update service ledger',size=10,weight='bold')
text(ax,1.64,.42,r'$q_{t+1}=q_t+a_t-x_t$',size=11)
box(ax,3.77,.20,3.26,.73,C['white'],C['axis'])
text(ax,5.40,.72,'Settle the common financial account',size=9.6,weight='bold')
text(ax,5.40,.42,'Electricity cash flow + terminal obligations',size=9.2)
elbow(ax,[(5.54,1.37),(5.54,1.15),(1.64,1.15),(1.64,.93)],C['axis'])
arrow(ax,(6.44,1.37),(6.44,.93),C['axis'])
arrow(ax,(.61,.93),(.61,1.37),C['purple'])
text(ax,.90,1.13,'Next step',size=9.0,ha='left',color=C['axis'])
text(ax,3.48,1.16,'Executed service',size=9.0,va='bottom',color=C['axis'])
export(fig,'figure1_worldmoe_main_readable')

fig,ax=setup(4.0)
text(ax,.15,3.78,'Contract rights and the value objects they identify',size=12,
     weight='bold',ha='left')
box(ax,.15,3.17,6.90,.40,C['pale'],C['grid'])
text(ax,3.60,3.37,'Hold fixed: admissions, realised prices, physics, baseline and terminal account',size=9.2)

text(ax,.15,2.94,'(a) Rights and their certified premium',size=10.0,weight='bold',ha='left')
# Nested rectangles depict sets, not measured areas.
box(ax,.23,2.11,3.17,.60,C['pale'],C['pink'])
text(ax,2.60,2.41,r'Wider rights  $\mathcal{X}_B$',size=10.0)
box(ax,.39,2.22,1.17,.37,C['lavender'],C['purple'])
text(ax,.975,2.405,r'$\mathcal{X}_A$',size=11)
text(ax,1.86,1.90,r'$\mathcal{X}_A\subseteq\mathcal{X}_B$',size=11.5)

text(ax,.28,1.48,'A',size=10,weight='bold',ha='left')
text(ax,.28,.91,'B',size=10,weight='bold',ha='left')
ax.plot([.62,1.44],[1.48,1.48],color=C['purple'],lw=4,solid_capstyle='butt')
ax.plot([2.18,3.21],[.91,.91],color=C['pink'],lw=4,solid_capstyle='butt')
for x,y,col in [(.62,1.48,C['purple']),(1.44,1.48,C['purple']),
                 (2.18,.91,C['pink']),(3.21,.91,C['pink'])]:
    ax.plot([x,x],[y-.10,y+.10],color=col,lw=1.1)
text(ax,.62,1.70,r'$H_A$',size=10.5)
text(ax,1.44,1.70,r'$U_A$',size=10.5)
text(ax,2.18,.69,r'$H_B$',size=10.5)
text(ax,3.21,.69,r'$U_B$',size=10.5)
arrow(ax,(1.44,1.19),(2.18,1.19),C['axis'],.95,'<->',7)
text(ax,2.35,1.48,r'$H_B-U_A>0$',size=11,weight='bold')
text(ax,1.82,.30,'Feasible value H; certified upper bound U\nSchematic positions, no numeric scale',size=9.0,color=C['axis'])

ax.plot([3.62,3.62],[.23,2.99],color=C['grid'],lw=1)
text(ax,3.84,2.94,'(b) Three distinct value objects',size=10.0,weight='bold',ha='left')
text(ax,3.89,2.52,r'Policy saving $S_W$',size=10.3,weight='bold',ha='left',color=C['purple'])
text(ax,3.89,2.27,'One verified operating trajectory',size=9.2,ha='left')
ax.plot([3.89,7.02],[2.06,2.06],color=C['grid'],lw=.8)
text(ax,3.89,1.83,r'PI opportunity $V_W^{PI}\in[H_W,U_W]$',size=10.3,weight='bold',ha='left',color=C['pink'])
text(ax,3.89,1.56,'Best feasible use of the historical path',size=9.2,ha='left')
ax.plot([3.89,7.02],[1.35,1.35],color=C['grid'],lw=.8)
text(ax,3.89,1.10,r'Prospective value $V_W^{\mathcal{F}}$',size=10.3,weight='bold',ha='left')
text(ax,3.89,.83,'Adapted policy and expected future outcomes',size=9.2,ha='left')
text(ax,3.89,.34,'A prospective contract price also needs\ninformation and tenant-cost evidence.',size=9.0,ha='left',color=C['axis'])
export(fig,'figure_contract_value_identification')

layout={'canvas_inches':[7.2,4.0],'recommended_main_width_inches':6.5,
 'minimum_designed_font_pt':9.0,'minimum_at_recommended_width_pt':9.0*6.5/7.2,
 'palette':C,'mechanism':{'input_boxes':[.14,1.37,1.40,1.79],
 'forecast_box':[1.91,2.45,2.91,.77],'valuation_box':[1.91,1.37,2.91,.80],
 'executor_box':[5.21,1.37,1.82,1.55],'ledger_box':[.14,.20,3.0,.73],
 'account_box':[3.77,.20,3.26,.73]},
 'facts_preserved':['Frozen W0/W1 sparse MoE market representation',
   'R2/APIC price distributions','Physical and queue-dependent valuation',
   'Original release/deadline and capacity executor','Current 15-minute action only',
   'Persistent service ledger and common terminal financial account'],
 'conceptual_figure':'Pure symbols. No 12/24 or seven-window numeric certificate version is embedded.',
 'scope':'Main-text mechanism simplification; detailed original architecture remains unchanged in supplement.'}
(OUT/'concept_figure_layout.json').write_text(json.dumps(layout,indent=2),encoding='utf-8')
print('Created readable main mechanism and symbolic contract-identification figure.')
