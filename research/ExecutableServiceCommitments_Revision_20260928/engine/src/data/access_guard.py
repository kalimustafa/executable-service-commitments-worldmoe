"""Explicit target-date guards for separate training and selection datasets."""
from pathlib import Path
from src.runtime import ROOT,WORKSPACE
import pandas as pd

def install_guard():
    if not ROOT.is_relative_to(WORKSPACE):raise RuntimeError('Invalid task output directory')

def assert_years(values,role='development'):
    dates=pd.to_datetime(values,utc=True,errors='raise')
    years=dates.year if hasattr(dates,'year') else dates.dt.year
    if bool((years>2024).any()):raise RuntimeError(f'Future evaluation data in {role}')
    return dates

def read_development_parquet(path,columns=None):
    path=Path(path)
    frame=pd.read_parquet(path,columns=columns)
    column='date' if 'date' in frame else 'timestamp_utc' if 'timestamp_utc' in frame else None
    if column is None:raise RuntimeError('Target date required when loading model development data')
    assert_years(frame[column],str(path))
    return frame

def assert_target_year(year):
    if int(year)>2024:raise RuntimeError('Training/selection runner cannot read final-year targets')

def register_historical(path):return Path(path)

