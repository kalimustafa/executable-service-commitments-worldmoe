"""Assemble formal bilingual manuscripts only from completed annual results."""
from pathlib import Path
import re,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT.parent;M=ROOT/'manuscript'
NAMES={'Published_DA':'Published DA','Old_RASC':'RASC','R2_full_native':'R2','Old_R_action':'Old R'}
def table(frame):
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'
def money(v):return f'US${v:,.2f}' if v>=0 else f'US$−{abs(v):,.2f}'
def main():
    status=json.loads((ROOT/'results/analysis_completed.json').read_text());assert status['status']=='complete','Wait for the complete matched neural baseline'
    f=pd.read_csv(ROOT/'results/policy_comparison_2025.csv');g=f[f.window_h.eq(12)&f.primary_comparison].copy()
    old=g[g.model.eq('Old_R_action')].iloc[0]
    pieces={}
    for lang in ['en','cn']:
        body=(M/f'main_{lang}.md').read_text(encoding='utf-8')
        body=body.replace('{{NEURAL_VALUE}}',money(old.value_usd)).replace('{{NEURAL_SHIFT}}',f'{old.shifted_IT_mwh:,.3f}')
        order={'Old_R_action_native':0,'Published_DA_point':1,'Old_RASC_point':2,'Old_RASC_robust':3,'R2_full_native_point':4,'R2_full_native_robust':5}
        g['order']=(g.model+'_'+g.policy).map(order);g=g.sort_values('order')
        label=g.model.map(NAMES)+' '+g.policy.map({'native':'action','point':'point','robust':'interval'})
        if lang=='cn':label=label.str.replace('Published DA','已公布DA',regex=False).str.replace('Old R action','旧R动作',regex=False).str.replace(' point','点预测',regex=False).str.replace(' interval','区间',regex=False)
        headers=['Policy','Value US$','Shifted IT MWh','CVaR95 US$','Late IT MWh'] if lang=='en' else ['策略','价值 US$','调节 IT MWh','CVaR95 US$','逾期 IT MWh']
        vals=[label,g.value_usd.map(lambda x:f'{x:,.2f}'),g.shifted_IT_mwh.map(lambda x:f'{x:,.3f}'),g.mtm_daily_loss_cvar95_usd.map(lambda x:f'{x:,.2f}'),g.late_IT_mwh.map(lambda x:f'{x:,.3f}')]
        body=body.replace('{{TABLE_MAIN}}',table(pd.DataFrame(dict(zip(headers,vals)))))
        methods=(M/f'methods_{lang}.md').read_text(encoding='utf-8')
        pieces[lang]=(body,methods)
    # Abstract is written after main results, discussion and methods are complete.
    abstract_en='''Computing loads can change their electricity use, but market value depends on whether the resulting work obligations remain executable. Here we connect frozen world representations and coherent price distributions to continuous reservations with deadline-priority recourse. A four-site 2025 replay establishes a 6.5-hour physical service floor. At 12 hours, the R2 interval policy earns US$31,219, retaining 76.8% of its point-policy value with 41.6% of the shifted energy and 42.8% lower daily-loss CVaR. At 24 hours, R2 point guidance adds US$3,990 over the existing forecast comparator, with a paired within-year 95% interval of US$1,832–6,297. For mixed 4/12/24-hour commitments, recourse eliminates 5.463 MWh of late work at a US$3.12 value change. These results distinguish the economic contributions of execution, prediction and risk selection, establishing service commitments as an operating basis for computing flexibility.'''
    abstract_cn='''算力负荷可以改变用电时间，但市场价值取决于由此形成的工作义务能否实际完成。本文将冻结世界表征和一致价格分布与连续预约、截止时间优先重排连接。在四站点完整 2025 年回放中，我们得到 6.5 小时物理服务窗口下限。12 小时下，R2 区间策略获得 US$31,219，以点预测策略 41.6% 的调节电量保留 76.8% 的价值，并使日损失 CVaR 降低 42.8%。24 小时下，R2 点预测较既有预测对照增加 US$3,990，年度内配对 95% 区间为 US$1,832–6,297。混合 4/12/24 小时承诺下，重排消除 5.463 MWh 逾期工作，价值仅变化 US$3.12。结果区分了执行、预测和风险筛选的经济贡献，确立了以服务承诺组织算力柔性的运行基础。'''
    refs=(PAPER/'revision_20260920_formal_bilingual/manuscript/references.md').read_text(encoding='utf-8')
    refs+='\n17. Gneiting, T. & Raftery, A. E. Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association* **102**, 359–378 (2007). [https://doi.org/10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).\n18. Elmachtoub, A. N. & Grigas, P. Smart “Predict, then Optimize”. *Management Science* **68**, 9–26 (2022). [https://doi.org/10.1287/mnsc.2020.3922](https://doi.org/10.1287/mnsc.2020.3922).\n'
    # Renumber once by first appearance in the English article; preserve all
    # cross-language and supplementary citation identities.
    order=[]
    for match in re.finditer(r'\^([0-9,–-]+)\^','\n'.join(pieces['en'])):
        for part in match.group(1).split(','):
            vals=list(map(int,re.split('[–-]',part)));ids=range(vals[0],vals[-1]+1)
            for n in ids:
                if n not in order:order.append(n)
    mapping={n:i+1 for i,n in enumerate(order)}
    def cites(s):
        def one(m):
            ids=[]
            for part in m.group(1).split(','):
                nums=list(map(int,re.split('[–-]',part)));ids.extend(range(nums[0],nums[-1]+1))
            return '^'+','.join(str(n) for n in sorted({mapping[n] for n in ids}))+'^'
        return re.sub(r'\^([0-9,–-]+)\^',one,s)
    records={int(a):b for a,b in re.findall(r'^([0-9]+)\. (.*)$',refs,flags=re.M)}
    reftext='\n'.join(f'{mapping[n]}. {records[n]}' for n in order)
    for lang,abstract in [('en',abstract_en),('cn',abstract_cn)]:
        body,methods=pieces[lang];title,rest=body.split('\n',1)
        assembled=title+'\n\n'+('## Abstract' if lang=='en' else '## 摘要')+'\n\n'+abstract+'\n\n'+rest.strip()+'\n\n'+methods.strip()
        assembled=cites(assembled)+'\n\n'+('## References' if lang=='en' else '## 参考文献')+'\n\n'+reftext+'\n'
        if lang=='cn':assembled+='\n'+cites((M/'supplement_cn.md').read_text(encoding='utf-8'))
        name='NatureEnergy_ExecutableFlexibility_R2_EN' if lang=='en' else 'NatureEnergy_ExecutableFlexibility_R2_CN'
        (M/(name+'.md')).write_text(assembled,encoding='utf-8')
    supp=cites((M/'supplement_en.md').read_text(encoding='utf-8'))
    (M/'Supplementary_ExecutableFlexibility_R2_EN.md').write_text(supp+'\n## References\n\n'+reftext+'\n',encoding='utf-8')
    abstract_words=len(re.findall(r"\b[\w]+(?:[’'–-][\w]+)*\b",abstract_en));assert abstract_words<=150
    body_only='\n'.join(x for x in pieces['en'][0].split('\n\n') if not x.startswith(('![','**Fig.','**Table','|','# ')))
    main_words=len(re.findall(r"\b[\w]+(?:[’'–-][\w]+)*\b",body_only));assert main_words<=3000
    manifest=dict(main_words_excluding_legends_tables=main_words,abstract_words=abstract_words,main_figures=5,main_tables=1,reference_count=len(order),supplement_tables=19,all_portrait=True,neural_evidence_complete=True)
    (M/'manuscript_structure.json').write_text(json.dumps(manifest,indent=2))
    (M/'abstract_en.md').write_text(abstract_en,encoding='utf-8');(M/'abstract_cn.md').write_text(abstract_cn,encoding='utf-8')
    (M/'references.md').write_text('## References\n\n'+reftext,encoding='utf-8')
    print(manifest)
if __name__=='__main__':main()
