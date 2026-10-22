"""Issue-independent observer inputs, with every observation strictly lagged."""
import numpy as np
import pandas as pd
from src.runtime import ROOT, LEGACY, WORKSPACE
from src.data.solar_nowcast import calendar_index


def observer_inputs(times, observations):
    times = pd.DatetimeIndex(times)
    pv = np.asarray(observations, np.float32)
    assert pv.shape == (len(times), 4)
    with np.load(ROOT / 'cache/solar_seasonal_envelope.npz') as z:
        high = z['upper'][calendar_index(times), times.hour * 4 + times.minute // 15]
    with np.load(ROOT / 'cache/causal_observer.npz') as z:
        alpha = z['alpha']
        residual = z['residual_quantiles'][times.hour * 4 + times.minute // 15]
    valid = np.isfinite(pv)
    safe = np.nan_to_num(pv, nan=0.)
    previous = np.vstack([np.zeros((1, 4), np.float32), safe[:-1]])
    previous_high = np.vstack([np.zeros((1, 4), np.float32), high[:-1]])
    scaled = previous * np.clip(high / np.maximum(previous_high, .02), 0, 2)
    point = np.clip(previous + alpha * (scaled - previous), 0, 1.5)
    base = np.clip(point[:, :, None] + residual * np.maximum(high[:, :, None], .1), 0, 1.5)
    normalized = safe / np.maximum(high, .1)
    innovation = (safe - point) / np.maximum(high, .1)
    local = np.stack([np.clip(normalized, 0, 3), high, safe,
                      np.clip(innovation, -2, 2), valid.astype(np.float32)], -1)
    # All indices are negative relative to the decision origin; current PV is
    # retained only as a supervised target, never as an input to its own row.
    padded = np.pad(local, ((16, 0), (0, 0), (0, 0)))
    histories = np.stack([padded[j:j+len(times)] for j in range(16)], axis=2)
    tday = (times.hour.to_numpy() + times.minute.to_numpy() / 60) / 24
    season = calendar_index(times) / 366
    clock = np.stack([np.sin(2*np.pi*tday), np.cos(2*np.pi*tday),
                      np.sin(2*np.pi*season), np.cos(2*np.pi*season)], -1)
    previous_valid = np.pad(valid.all(-1), (16, 0))
    history_valid = np.stack([previous_valid[j:j+len(times)] for j in range(16)], -1).all(-1)
    return dict(history=histories.astype(np.float32), base_quantiles=base.astype(np.float32),
                upper=high.astype(np.float32), clock=clock.astype(np.float32),
                target=safe.astype(np.float32), valid=valid.all(-1)&history_valid)


def training_data():
    with np.load(LEGACY / 'cache/development_physical_reference.npz') as z:
        dates=pd.DatetimeIndex(pd.to_datetime(z['dates'], utc=True))
        keep=dates.year<=2023;dates=dates[keep];pv=z['pv'][keep]
    assert set(dates.year)=={2022, 2023}
    times=pd.date_range('2022-01-01','2024-01-01',freq='15min',inclusive='left',tz='UTC')
    observations=np.full((len(times),4),np.nan,np.float32)
    for date,value in zip(dates,pv):
        i=int((date-times[0]).total_seconds()/900);observations[i:i+96]=value
    data=observer_inputs(times,observations)
    return {key:value[data['valid']] for key,value in data.items() if key!='valid'}


def annual_data(year):
    with np.load(WORKSPACE / f'external_validation/data/forcing_{year}.npz') as z:
        times=pd.DatetimeIndex(pd.to_datetime(z['times'],utc=True));pv=z['pv']
    return observer_inputs(times,pv)
