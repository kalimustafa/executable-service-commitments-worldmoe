import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.context import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--year',type=int,default=2025);p.add_argument('--seed',type=int,default=20260920);args=p.parse_args()
rows=[]
for window in [12,16,24]:
    for post in [False,True]:rows.append(run_persistent(args.year,'Eagle',args.seed,window,post))
pd.DataFrame(rows).to_csv(RESULTS/f'information_{args.year}_{args.seed}.csv',index=False)
