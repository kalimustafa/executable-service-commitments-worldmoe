"""Fail closed on locked development years and writes to the original workspace.

Run dates and seeds are not target years. A narrowly registered historical
artifact can reside in an old directory named *2025final; its date contract
must still be checked before loading targets. New experiments have no unlock.
"""
from pathlib import Path
import os
import re
import sys
from src.runtime import ROOT, WORKSPACE, REPO

_installed = False
_allowed_historical = set()
_locked = re.compile(r'(?<!\d)20(?:2[5-9]|[3-9]\d)(?!\d)')

def assert_years(values, role='development'):
    import pandas as pd
    dates = pd.to_datetime(values, utc=True, errors='raise')
    if getattr(dates, 'hasnans', False):
        raise RuntimeError(f'Missing timestamps in {role}')
    years = dates.year if hasattr(dates, 'year') else dates.dt.year
    if bool((years > 2024).any()):
        raise RuntimeError(f'LOCKED YEAR: {role} requires target_year <= 2024')
    return dates

def assert_target_year(target_year):
    if int(target_year) > 2024:
        raise RuntimeError('LOCKED YEAR: target_year must be <= 2024')

def register_historical(path):
    """Only call for reviewed W1 weights/metadata or a date-checked 2024 file."""
    _allowed_historical.add(str(Path(path).resolve()).casefold())

def checked_path(path, *, write=False):
    p = Path(path).resolve()
    if p.is_relative_to(ROOT):
        if not write and _locked.search(p.name) and p.suffix.lower() in {'.npz','.npy','.pt','.ckpt','.parquet','.csv'}:
            raise RuntimeError(f'LOCKED DATA PATH: {p}')
        return p
    if write and p.is_relative_to(WORKSPACE):
        raise RuntimeError(f'READ ONLY ORIGINAL PROJECT: {p}')
    if not write and p.is_relative_to(WORKSPACE):
        if p.is_relative_to(REPO / '.venv' / 'Lib' / 'site-packages'):
            return p  # Installed package versions are not development years.
        if p.suffix.lower() in {'.py', '.pyc', '.pyd', '.dll', '.yaml', '.yml'}:
            return p
        if _locked.search(str(p)) and str(p).casefold() not in _allowed_historical:
            raise RuntimeError(f'LOCKED DATA PATH: {p}')
    return p

def _audit(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        mode = args[1] if len(args) > 1 else ''
        flags = args[2] if len(args) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        checked_path(os.fsdecode(args[0]), write=writing)
    elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod'}:
        if args and isinstance(args[0], (str, bytes, os.PathLike)):
            checked_path(os.fsdecode(args[0]), write=True)
    elif event in {'os.rename', 'os.replace'}:
        for value in args[:2]:
            if isinstance(value, (str, bytes, os.PathLike)):
                checked_path(os.fsdecode(value), write=True)

def install_guard():
    global _installed
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True
        # Native pandas/Arrow/NumPy readers do not consistently emit Python
        # open audit events. Guard their public data entrypoints as well.
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq
        import torch
        original_np_load=np.load
        original_read_parquet=pd.read_parquet
        original_to_parquet=pd.DataFrame.to_parquet
        original_torch_load=torch.load

        def guarded_np_load(file,*args,**kwargs):
            if isinstance(file,(str,bytes,os.PathLike)):checked_path(os.fsdecode(file))
            value=original_np_load(file,*args,**kwargs)
            if isinstance(value,np.lib.npyio.NpzFile) and 'dates' in value.files:
                try:assert_years(value['dates'],'NumPy target dates')
                except BaseException:
                    value.close();raise
            return value

        def guarded_read_parquet(path,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):
                checked_path(os.fsdecode(path))
                # Read temporal metadata/columns first. For the mixed-year
                # weather store, retain its formal <=2024 predicate pushdown.
                names=pq.read_schema(path).names
                temporal='date' if 'date' in names else ('timestamp_utc' if 'timestamp_utc' in names else None)
                if temporal:
                    probe=dict(kwargs);probe['columns']=[temporal]
                    dates=original_read_parquet(path,**probe)[temporal]
                    assert_years(dates,f'Parquet {temporal}: {path}')
            return original_read_parquet(path,*args,**kwargs)

        def guarded_to_parquet(frame,path=None,*args,**kwargs):
            if isinstance(path,(str,bytes,os.PathLike)):checked_path(os.fsdecode(path),write=True)
            return original_to_parquet(frame,path,*args,**kwargs)

        def guarded_torch_load(file,*args,**kwargs):
            if isinstance(file,(str,bytes,os.PathLike)):checked_path(os.fsdecode(file))
            return original_torch_load(file,*args,**kwargs)

        np.load=guarded_np_load
        pd.read_parquet=guarded_read_parquet
        pd.DataFrame.to_parquet=guarded_to_parquet
        torch.load=guarded_torch_load

def read_development_parquet(path, *, date_column='date', columns=None):
    import pandas as pd
    # Open only the temporal column before any target/feature column.
    p = checked_path(path)
    dates = pd.read_parquet(p, columns=[date_column])[date_column]
    assert_years(dates, str(p))
    return pd.read_parquet(p, columns=columns)
