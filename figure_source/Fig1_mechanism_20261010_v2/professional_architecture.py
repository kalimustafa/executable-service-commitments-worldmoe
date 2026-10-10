"""Compact publication-style WorldMoE architecture, drawn entirely in Python.

The expert highlights are schematic routing examples, not measured assignments.
All coloured fills use the supplied lavender/coral palette or its white tints.
The APIC market-price branch bypasses queue conditioning and reaches the cost
decoder separately. Source: Supplementary Table 16; coupled_rollout.py;
continuous_decision.py; vpp_world_moe/config.py and sparse_moe.py.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgb, to_hex

PALETTE = ['#B6B3D6', '#CFCCE3', '#D5D3DE', '#D5D1D1',
           '#F6DFD6', '#F8B2A2', '#F1837A', '#E9687A']
INK = '#3E3C43'
WHITE = '#FFFFFF'

def _tint(colour, strength):
    rgb = to_rgb(colour)
    return to_hex(tuple(1 - strength * (1 - c) for c in rgb))

def draw_worldmoe_architecture(ax, x, y, width, height):
    """Draw in an existing data-coordinate bounding box; return all artists.

    Intended physical size is 5.97 x 4.10 inches. Labels use 11.5--13 pt,
    with 12 pt as the standard module font and 12.6 pt group headings.
    The function does not create axes, modify axis limits or change rcParams.
    """
    sx, sy = width / 5.97, height / 4.10
    artists = []

    def xy(px, py):
        return (x + sx * px, y + sy * py)

    def text(px, py, label, size=12, bold=False, ha='center', **kwargs):
        a = ax.text(*xy(px, py), label, ha=ha, va='center', fontsize=size,
                    fontweight='bold' if bold else 'normal', color=INK,
                    linespacing=1.13, zorder=25, **kwargs)
        artists.append(a)
        return a

    def rect(px, py, w, h, fill=WHITE, edge=INK, lw=.78, radius=0, z=12):
        if radius:
            a = FancyBboxPatch(xy(px, py), w*sx, h*sy,
                              boxstyle=f'round,pad=0,rounding_size={radius*min(sx,sy)}',
                              facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z)
        else:
            a = Rectangle(xy(px, py), w*sx, h*sy, facecolor=fill,
                          edgecolor=edge, linewidth=lw, zorder=z)
        ax.add_patch(a); artists.append(a)
        return a

    def node(px, py, w, h, label, fill=WHITE, size=12, bold=False):
        a = rect(px, py, w, h, fill=fill)
        text(px+w/2, py+h/2, label, size=size, bold=bold)
        return a

    def wire(points, arrow=True, dashed=False, lw=1.0):
        # Explicit orthogonal segments give repeatable bends and avoid routing
        # through other nodes. Last segment alone receives the arrowhead.
        if len(points) > 2:
            a, = ax.plot(*zip(*(xy(*p) for p in points[:-1])), color=INK,
                         linewidth=lw, linestyle=(0,(2.2,1.6)) if dashed else '-',
                         zorder=18, solid_capstyle='butt')
            artists.append(a)
        end = FancyArrowPatch(xy(*points[-2]), xy(*points[-1]),
                              arrowstyle='-|>' if arrow else '-', mutation_scale=8.5,
                              linewidth=lw, color=INK, shrinkA=0, shrinkB=0,
                              linestyle=(0,(2.2,1.6)) if dashed else '-', zorder=18)
        ax.add_patch(end); artists.append(end)

    def group(px, w, title, colour):
        rect(px,.025,w,4.035,fill=_tint(colour,.19),edge=_tint(colour,.80),lw=.9,radius=.045,z=3)
        rect(px+.001,3.61,w-.002,.45,fill=_tint(colour,.66),edge='none',z=4)
        text(px+w/2,3.837,title,size=12.6,bold=True)

    group(.02,1.78,'Frozen W0/W1',PALETTE[0])
    group(2.03,1.88,'Market–physics',PALETTE[2])
    group(4.14,1.81,'Value decoding',PALETTE[4])

    # Stage 1. The ten-block frozen world backbone, with the sparse sublayer
    # expanded into shared, general Top-2, and gated-tail parallel paths.
    node(.17,3.02,1.48,.43,'10 interaction\nblocks',_tint(PALETTE[0],.45))
    rect(.15,.96,1.52,1.94,fill=WHITE,edge=_tint(PALETTE[0],.85),lw=.85,z=5)
    text(.91,2.73,'Sparse MoE ×5',12.0,True)
    wire([(.91,3.02),(.91,2.90)])
    wire([(.91,2.55),(.24,2.55),(.24,1.27)],arrow=False)
    wire([(1.58,2.32),(1.58,1.27)],arrow=False)
    node(.35,2.155,1.12,.33,'Shared',_tint(PALETTE[1],.65),12.0)
    node(.35,1.545,1.12,.46,'Top-2 / 7',_tint(PALETTE[1],.24),11.8)
    # Place the label in the upper part of the routed module and seven
    # aligned expert cells underneath; two arbitrary cells are highlighted.
    artists[-1].set_position(xy(.91,1.867))
    for j in range(7):
        rect(.409+j*.143,1.618,.108,.118,
             fill=PALETTE[6] if j in (1,4) else PALETTE[1],
             edge=INK,lw=.43,z=22)
    node(.35,1.105,1.12,.33,'Gated tail',_tint(PALETTE[5],.40),11.7)
    for yy in (2.32,1.775,1.27):
        dashed = yy == 1.27
        wire([(.24,yy),(.35,yy)],dashed=dashed,lw=.8)
        wire([(1.47,yy),(1.58,yy)],arrow=False,dashed=dashed,lw=.8)
    wire([(1.58,1.27),(1.58,.82),(.91,.82),(.91,.69)])
    node(.17,.24,1.48,.45,'Frozen features\n+ R2 cache',_tint(PALETTE[0],.38),11.8)

    # The frozen evidence is reused in the downstream decision module.
    # Route this connector in the reserved inter-group gutter.
    wire([(1.65,.465),(1.915,.465),(1.915,3.235),(2.19,3.235)],lw=1.05)

    # Stage 2. APIC precedes every queue-dependent operation. Its price output
    # has a dedicated protected bypass in the next stage.
    node(2.19,3.02,1.55,.43,'APIC adapter',_tint(PALETTE[5],.50),12.2)
    wire([(3.74,3.235),(4.29,3.235)],lw=1.05)
    node(2.14,2.28,.78,.49,'Market\nencoder',_tint(PALETTE[1],.56),11.9)
    node(3.02,2.28,.78,.49,'Physics\nencoder',_tint(PALETTE[3],.55),11.9)
    wire([(2.565,3.02),(2.565,2.90),(2.53,2.90),(2.53,2.77)])
    text(3.41,2.945,'PV · PUE',11.5)
    wire([(3.41,2.85),(3.41,2.77)],lw=.8)
    node(2.22,1.58,1.51,.47,'Cross-attention\n4 heads · bidir.',_tint(PALETTE[0],.42),11.7)
    wire([(2.53,2.28),(2.53,2.05)])
    wire([(3.41,2.28),(3.41,2.05)])
    node(2.22,.77,1.51,.48,'Temporal ×2\n24 hourly tokens',_tint(PALETTE[0],.50),11.7)
    wire([(2.975,1.58),(2.975,1.25)])

    # Stage 3. The protected quantiles and queue-conditioned value branch
    # meet only at the cost decoder. No arrow returns from queue to APIC.
    node(4.29,3.02,1.47,.43,'7 price quantiles',_tint(PALETTE[5],.50),11.65)
    wire([(5.76,3.235),(5.855,3.235),(5.855,.47),(5.64,.47)],lw=1.0)
    node(4.29,2.34,1.35,.47,'Queue queries\n96 slots',_tint(PALETTE[4],.77),12.0)
    node(4.29,1.65,1.35,.47,'Work-to-time\nattention',_tint(PALETTE[1],.56),11.9)
    wire([(4.965,2.34),(4.965,2.12)])
    wire([(3.73,1.01),(4.025,1.01),(4.025,1.885),(4.29,1.885)],lw=1.05)
    node(4.29,.96,1.35,.43,'Slot refinement',_tint(PALETTE[0],.46),11.7)
    wire([(4.965,1.65),(4.965,1.39)])
    node(4.29,.22,1.35,.50,'Cost heads\n+ opportunity',_tint(PALETTE[6],.47),12.0)
    wire([(4.965,.96),(4.965,.72)])
    wire([(4.965,.22),(4.965,.055)],lw=1.05)
    return artists


if __name__ == '__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    fig = plt.figure(figsize=(5.97,4.10),facecolor=WHITE)
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,5.97),ylim=(0,4.10)); ax.axis('off')
    draw_worldmoe_architecture(ax,0,0,5.97,4.10)
    destination=Path(__file__).resolve().parent
    for suffix in ('png','pdf','svg'):
        fig.savefig(destination/f'professional_architecture_preview.{suffix}',dpi=240)
    plt.close(fig)
