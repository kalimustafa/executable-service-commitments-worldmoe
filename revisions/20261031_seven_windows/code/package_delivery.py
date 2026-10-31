from pathlib import Path
import shutil,json,hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'EnergyEconomics_Revision_20261031'
PKG=ROOT/'publication_package';PKG.mkdir(exist_ok=True);OUT=ROOT/'deliverables'
ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo','*.nbc','*.nbi','*.lock','*_template.csv','manuscript_framework.txt')
for folder in ['code','figures','results','audit','certificates','analysis','frozen_runtime','inputs','literature','logs']:
    shutil.copytree(ROOT/folder,PKG/folder,dirs_exist_ok=True,ignore=ignore)
shutil.copytree(OLD/'publication_package',PKG/'prior_evidence',dirs_exist_ok=True,ignore=ignore)
(PKG/'manuscript').mkdir(exist_ok=True)
for p in OUT.glob('*.docx'):shutil.copy2(p,PKG/'manuscript'/p.name)
for name in ['main.md','reference_number_map.json','S32_results_appendix.md']:
    shutil.copy2(ROOT/'manuscript'/name,PKG/'manuscript'/name)
p=PKG/'manuscript/main.md';p.write_text(p.read_text(encoding='utf-8').replace('(figures/','(../figures/'),encoding='utf-8')
for p in (ROOT/'manuscript/figures').iterdir():
    if p.is_file():shutil.copy2(p,PKG/'figures'/p.name)
for name in ['README.md','README_RUNTIME.md','PROTOCOL.md','S32_window_extension_methods.md','frozen_source_dependencies.json','policy_completion.json','suite_status.json']:
    shutil.copy2(ROOT/name,PKG/name)
# Refresh packaging-only file hashes after adding the final explanatory notes.
cm=PKG/'certificates/package_manifest.json'
if cm.exists():
    data=json.loads(cm.read_text(encoding='utf-8'))
    for item in data.get('files',[]):
        fp=cm.parent/item['path']
        if fp.exists():item.update(bytes=fp.stat().st_size,sha256=hashlib.sha256(fp.read_bytes()).hexdigest())
    cm.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
(PKG/'requirements-analysis.txt').write_text('numpy==2.1.3\nscipy==1.15.3\nmatplotlib>=3.9\n',encoding='utf-8')
(PKG/'requirements-replay.txt').write_text('numpy==2.1.3\npandas==2.2.3\nnumba==0.61.0\ntorch==2.9.0\npyarrow>=18\n',encoding='utf-8')
report='''2026年10月31日七窗口扩展与论文修订

一、实验已完成
15组新全年策略运行：8/20/36/48小时×三个原检查点，加三组24小时锚点。三组24小时完整路径逐位复现历史结果。没有新训练或调参。
七窗口实际准入证书：8/12/16/20/24/36/48小时×四站，共28组，均保留完整exact witness、乘子和原非凸成本结算。
8小时三个策略均出现约3.01 IT MWh实质逾期，原始现金结果明确标为履约失败，未用数值修复掩盖。其他12组新运行给出独立exact可行数值见证，最大货币修复界小于1.10e-8美元。

二、更新的价值识别
实际准入24小时PI价值：US$132376.82–132598.65。
24小时剩余机会：US$27105.76–27327.58；捕获率向外舍入79.39%–79.53%。
最优12至24小时权利价值差：US$34580.41–34959.31，与历史策略差US$33050.09分开。
24至36小时PI每增加期限小时783.16–823.30美元；36至48小时878.48–922.81美元，后者严格回升。
同区间策略增益每期限小时却从748.34降至737.51美元，说明控制器收益曲线不能替代合同机会曲线。
原统一forall a exists x结论与宽区间保留为独立稳健性证据，未被实际准入结果覆盖。

三、图件和文献
正文6图，重排WorldMoE机制图，新增合同价值识别和现金流瀑布图，更新实际准入机会图和七窗口响应图，全部沿用原紫粉配色。
正文6幅及补充12幅图注均缩短：正文最长25英文词，补充最长23英文词。
引言补回三篇Energy Economics相关研究的简洁批判综述，明确本文固定服务与结算条件下的可核验货币界限能力。
原43张补充表（包含上一轮新S30表）及12幅图像保持不变；新增S31与S32后补充材料47表。

四、交付
正文18页，补充材料65页；公式保持Word原生可编辑数学。
当前稿与完整证据在本包，prior_evidence是上一版原包，旧“未新增窗口”说明仅为历史状态。
README提供独立审计、证书验证、绘图和冻结策略重放命令。
'''
(OUT/'七窗口修订说明_20261031.txt').write_text(report,encoding='utf-8');(PKG/'REVISION_RESPONSE_ZH.txt').write_text(report,encoding='utf-8')
manifest={'revision_date':'2026-10-31','main_pages':18,'supplement_pages':65,'main_figures':6,'supplement_figures':12,
    'main_caption_max_words':25,'supplement_caption_max_words':23,'new_policy_evaluations':15,'new_policy_service_valid':12,'eight_hour_failures_retained':3,
    'actual_admission_site_window_certificates':28,'model_training_performed':False,'prior_revision_commit':'822aac606a47c6be2639371f90ef2444c8320267'}
(PKG/'revision_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
files=sorted(p for p in PKG.rglob('*') if p.is_file() and p!=PKG/'SHA256SUMS.txt')
(PKG/'SHA256SUMS.txt').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(PKG).as_posix() for p in files)+'\n',encoding='utf-8')
zp=OUT/'EnergyEconomics_Seven_Window_Complete_Evidence_20261031.zip'
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(PKG.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(PKG).as_posix())
print(json.dumps({'files':len(files)+1,'zip_bytes':zp.stat().st_size,'largest_file_bytes':max(p.stat().st_size for p in files)},indent=2))
