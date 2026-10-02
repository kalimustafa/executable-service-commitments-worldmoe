"""Update the original editable vector architecture to the frozen new R2."""
from pathlib import Path
import importlib.util,inspect,textwrap,json
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parent/'新论文_市场可交付灵活性_20260917'
path=PAPER/'revision_20260920_r2_robustness/figures/build_architecture.py'
spec=importlib.util.spec_from_file_location('retained_architecture',path)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
OUT=ROOT/'figures';a.OUT=OUT;a.b.OUT=OUT;b=a.b
card_colors=dict(N='#3C3947',BLUE='#B6B3D6',TEAL='#B6B3D6',PURPLE='#E9687A',AMBER='#F8B2A2',RED='#F1837A',CYAN='#CFCCE3',GREEN='#D5D1D1')
for key,value in card_colors.items():setattr(b,key,value);setattr(a,key,value)
asset_source=textwrap.dedent(inspect.getsource(b.Scene.image)).replace("OUT.parent.parent/'revision_20260918/figures/material_library/renders'",repr(str(PAPER))+" and Path("+repr(str(PAPER))+")/'revision_20260918/figures/material_library/renders'")
exec(asset_source,b.__dict__);b.Scene.image=b.image
upper_source=inspect.getsource(a.upper).replace("['R2 market','encoder']","['R2 market','residual']")
upper_source=upper_source.replace("['Cached states','Expert priors','Price features']","['Frozen states','Day profile','Gated residual']")
upper_source=upper_source.replace("['Price and risk','distributions']","['APIC price','coupling']")
upper_source=upper_source.replace("['DA / DART / RT','7 quantiles','3 market regimes']","['DA / D / RT','7 quantiles','Cost scores']")
exec(upper_source,a.__dict__)
distribution_source=inspect.getsource(a.distributions).replace("['4  Price','distributions']","['4  APIC price','coupling']")
distribution_source=distribution_source.replace("['DA + DART','7 quantiles']","['DA + spread D','7 quantiles']")
distribution_source=distribution_source.replace("['RT derived from','DA − DART']","['Coupled prices','RT = DA − D']")
exec(distribution_source,a.__dict__)
N=b.N;PURPLE=b.PURPLE;TEAL=b.TEAL;AMBER=b.AMBER;RED=b.RED

def encoder(s):
    b.detail(s,639,222,PURPLE,['3  R2 market','residual'])
    boxes=[
      (525,75,['Frozen W1 states','Blocks 3 / 6 / 10','Sites · experts · tail']),
      (622,76,['Market query','Cross-attention · 128','Expert routing priors']),
      (720,76,['Daily-profile attention','Width 64','DA masked before issue']),
      (818,76,['Gate × residual','Zero initial correction','Base path preserved']),
      (916,79,['Median + increment','logit corrections','Ordered DA and spread'])]
    for i,(y,h,labels) in enumerate(boxes):
        s.rect(649,y,202,h,b.tint(PURPLE,.96),'#B7A6D4',.8,4)
        s.multiline(750,y+22,labels,16.5,22,'TNR-B' if i==3 else 'TNR',max_width=190)
        if i<len(boxes)-1:s.arrow([(750,y+h+2),(750,y+h+18)],N,1.1,6)
    s.arrow([(862,714),(877,714)],N,1.2,6)

def decision(s):
    b.detail(s,1044,397,RED,['5  Causal cost and execution'])
    s.rect(1053,520,379,147,'#FFFFFF','#BDCDD8',.75,5)
    s.text(1065,545,'Anchored price-identity coupling',19,'TNR-B',max_width=356)
    s.rect(1066,556,353,44,'#F0F8F8','#83B8BE',.7,4)
    s.text(1078,576,'Pre-DA joint samples',17,'TNR-B')
    s.text(1078,596,'RT(m) = DA(m) − D(m)',17,max_width=327)
    s.rect(1066,611,353,44,'#F6F2FB','#BBA9D5',.7,4)
    s.text(1078,631,'Published DA anchor',17,'TNR-B')
    s.text(1078,650,'qRT(τ | DA) = DA − qD(1 − τ | DA)',17,max_width=328)
    s.rect(1053,682,379,160,'#F8F6FC','#B7A6D4',.75,5)
    s.text(1065,707,'Reservation-state cost head',19,'TNR-B',max_width=355)
    s.multiline(1242,734,['Issued PV paths + completed 4 h observations',
        'Booked load + newly released work',
        'Expected incremental PV use and grid import',
        'Point or interval cost ranking'],16.5,26,max_width=355)
    s.arrow([(1242,845),(1242,864)],N,1.1,6)
    s.rect(1053,868,379,145,'#F9FCFB','#A7C9BF',.75,5)
    s.text(1065,893,'Executable service commitments',19,'TNR-B',max_width=356)
    for i in range(10):s.rect(1070+i*34,908,29,20,b.tint(TEAL if i%3 else AMBER,.65),'#A8C3C6',.6,3)
    s.multiline(1242,952,['Continuous capacity reservations',
        'Deadline-priority recourse',
        'Work conservation · physical power balance'],17,25,max_width=355)

def main():
    OUT.mkdir(exist_ok=True);s=b.Scene();s.pdf.setTitle('Frozen R2 with causal reservation-state cost prediction')
    a.upper(s);b.draw_backbone(s);b.draw_sparse(s);encoder(s);a.distributions(s);decision(s)
    images,text=s.finish()
    for ext in ['png','svg','pdf']:(OUT/f'Figure4_materials_model.{ext}').replace(OUT/f'Fig1_new_R2_architecture.{ext}')
    (OUT/'Fig1_source.json').write_text(json.dumps(dict(source_reference=str(path),frozen_W1_parameters=59156961,
        R2_total_head_parameters=969395,base_equivalent_initialization=True,live_reservations_enter_cost_head=True,
        completed_PV_feedback_hours=4,primary_policy='24 h interval',source_materials=s.materials,
        APIC_full_name='anchored price-identity coupling',
        APIC_formula_source=str(ROOT/'vendor/r2_market.py'),
        APIC_naming_source=str(ROOT.parent/'NatureEnergy_R-ECMoE_投稿修订_20260829/round7_apic_naming_20260905/revise_apic.py'),
        spread_symbol='D',APIC_formula='pre-DA paired RT = DA - D; post-DA q_RT(tau|DA) = DA - q_D(1-tau|DA)',
        naming_confirmation='User explicitly requested restoring the earlier APIC name throughout the figure and manuscript',
        palette_source=str(ROOT/'configs/visual_palette.json')),indent=2))
    assert 'DART' not in (OUT/'Fig1_new_R2_architecture.svg').read_text(encoding='utf-8')
    print('Figure 1 complete',images,text)

if __name__=='__main__':main()
