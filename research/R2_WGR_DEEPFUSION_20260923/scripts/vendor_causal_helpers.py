from src.runtime import ROOT, LEGACY
import ast
import hashlib
import json

source=LEGACY/'experiment/prepare_frozen_cache.py'
text=source.read_text(encoding='utf-8')
tree=ast.parse(text)
selected=[]
for node in tree.body:
    if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in {'cutoff','generation','disable_inactive_slots','causal_weather','MarketContext','reduce_features'}:
        code=ast.get_source_segment(text,node)
        if node.name=='MarketContext':
            first=code.index('        if evaluation_panel is not None:')
            last=code.index('        self.frame=',first)
            code=code[:first]+"        if evaluation_panel is not None:\n            raise RuntimeError('This branch has no locked-evaluation input mode')\n"+code[last:]
        selected.append(code)
header='''"""Minimal causal helper copy; no original mains or evaluation data loaders."""
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
from src.runtime import ROOT, PAPER
from dcvpp.vpp_world_moe.data import (_dynamic_graph, WEATHER_FEATURES, NYISO_WEATHER_SITES,
    WEATHER_TEMPERATURE_CENTER_K, WEATHER_TEMPERATURE_SCALE_K, WEATHER_SHORTWAVE_SCALE_WM2)
TZ='America/New_York'
ZONES=['WEST','CAPITL','N.Y.C.','LONGIL']
INACTIVE=(1,9,17,25)
PRICE_PANEL=PAPER/'experiments/gate_hourly_panel.parquet'
'''
(ROOT/'vendor/causal_world.py').write_text(header+'\n\n'.join(selected)+'\n',encoding='utf-8')
(ROOT/'configs/vendor_manifest.json').write_text(json.dumps(dict(causal_helpers_source=str(source),causal_helpers_source_sha256=hashlib.sha256(text.encode()).hexdigest(),
    transformations=['AST-selected causal helpers only','original globals replaced with read-only source paths','evaluation_panel forbidden'],
    preserved=['disable_inactive_slots','causal_weather','MarketContext.overlay','reduce_features','cutoff','generation']),indent=2))
