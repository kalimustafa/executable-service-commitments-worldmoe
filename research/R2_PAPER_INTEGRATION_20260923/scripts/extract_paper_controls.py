"""Copy only unchanged executor-control functions; exclude all legacy mains."""
from src.runtime import ROOT,PAPER
import ast,hashlib,json

sources=[
    (PAPER/'analysis/reviewer_response_20260920/experiment/task_a_executor_ablation.py',['myopic_execute']),
    (PAPER/'revision_20260920_market_expansion/experiment/repair_reservations.py',['run','edf_reference'])]
header='''import numpy as np
import pandas as pd
from collections import defaultdict
from types import SimpleNamespace
from vendor import economic_core as c
from vendor import reservation as d
H=SimpleNamespace(c=c)
CLASSES=[(.25,4),(.5,12),(.25,24)]
'''
parts=[header];records=[]
for path,names in sources:
    source=path.read_text(encoding='utf-8');tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            text=ast.get_source_segment(source,node);parts.append(text+'\n')
            records.append(dict(function=node.name,path=str(path),sha256=hashlib.sha256(text.encode()).hexdigest()))
target=ROOT/'vendor/paper_controls.py';target.write_text('\n'.join(parts),encoding='utf-8')
(ROOT/'configs/paper_executor_sources.json').write_text(json.dumps(records,indent=2))
print('Copied unchanged myopic/recourse/EDF functions',len(records))
