from src.runtime import ROOT, PAPER, LEGACY
from src.data.access_guard import install_guard
install_guard()
import ast
import hashlib
import json

def extract(path,names):
    text=path.read_text(encoding='utf-8');tree=ast.parse(text)
    funcs=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(funcs)==len(names)
    return '\n\n'.join(funcs),dict(path=str(path),sha256=hashlib.sha256(text.encode()).hexdigest(),functions=sorted(names),function_code_unmodified=True)

folder=PAPER/'revision_20260920_market_expansion/experiment'
core,one=extract(folder/'run_screen.py',{'risk','cash','record'})
header='''"""Unmodified formal accounting functions; no legacy data-loading main."""
import numpy as np
import pandas as pd
from src.runtime import ROOT
N=0
DT=.25
FEE=1.
MIGRATION_COST=5.
DATA=ROOT/'results/economic_trajectories'
DATA.mkdir(exist_ok=True)
DAILY=[]
ROWS=[]
TRAJ={}
'''
(ROOT/'vendor/economic_core.py').write_text(header+core+'\n',encoding='utf-8')
schedule,two=extract(folder/'deepen_scheduling.py',{'reference','rank','schedule'})
(ROOT/'vendor/reservation.py').write_text('"""Exact formal reservation, capacity calendar and late-work handling."""\nimport numpy as np\nfrom vendor import economic_core as c\n'+schedule+'\n',encoding='utf-8')
provider,three=extract(LEGACY/'experiment/ef_forecast_provider.py',{'ForecastProvider'})
(ROOT/'vendor/forecast_provider.py').write_text("import numpy as np\nimport pandas as pd\nZONES=['WEST','CAPITL','N.Y.C.','LONGIL']\nTZ='America/New_York'\nTAU=(.05,.10,.25,.5,.75,.9,.95)\n"+provider+'\n',encoding='utf-8')
cal,four=extract(LEGACY/'experiment/calibrate_r2.py',{'NativeCalibration'})
(ROOT/'src/calibration/native_affine.py').write_text('"""Exact formal 24-parameter affine calibrator."""\nimport torch\nfrom torch import nn\n'+cal+'\n',encoding='utf-8')
(ROOT/'configs/economic_source_manifest.json').write_text(json.dumps(dict(sources=[one,two,three,four],changes='Only module globals/import destinations; extracted function bodies unchanged; no original mains imported.'),indent=2))
