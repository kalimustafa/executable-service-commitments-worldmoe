"""Header-grounded Frontier parsing and pre-declared shape normalisation.

Primary demand is submission-time requested nodes from field 20's explicit
frontier:excl:<nodes>x<cores> property. Field 7 is retained as a separate
resource-count sensitivity because it mixes node/core resource magnitudes.
No status, peak or day filters are applied, and runtime never weights demand.
"""
import re,json
import numpy as np,pandas as pd
from paths import *
FIELDS=['job_id','submit_time','wait_time','runtime','allocated_resource','cpu_time','ram_used','required_resource','walltime_limit','ram_requested','status','user_id','group_id','executable','queue','partition','previous_job','think_time','used_gpu','soft_walltime','required_properties']

def main():
    constraints=json.loads((PAPER/'revision_20260920_service_frontier/experiment/results/completed.json').read_text())['constraints']
    k=np.asarray(constraints['work_power_per_unit'])
    z=np.load(PAPER/'revision_20260918/annual_continuous_2025/data/annual_forcing_2025.npz')
    a=z['actual_21_hourly_pricing'][:,12]
    target=float(.25*a.sum()*k.sum()/4)
    # Freeze choices before any new economic result is computed.
    protocol=dict(primary_proxy='field 20 explicit requested-node count in frontier:excl:Nx112',sensitivity_proxy='field 7 recorded required resource count',
        field7_role='header allows nodes or cores; do not label it physical nodes',requested_resource_scaling='one full-calendar scalar, no clipping or outcome-based choice',
        target_variable_it_mwh=target,target_source='frozen Eagle-2025 workload and IT conversion',runtime_weighted=False,all_statuses_retained=True,
        time_pairing='map elapsed seconds from header UnixStartTime onto NYISO market-year local Jan 1 midnight; constant 5 h UTC phase translation retains every submission and interarrival',
        market_years=[2024,2025],windows_h=[7,12,24],mixed_classes=[[.25,4],[.5,12],[.25,24]],
        R2_external_evaluation=False,model_training=False,secondary_field7_scope='12 h mechanisms and mixed-deadline recourse',
        primary_2025_frozen_unchanged=True,publication_time='11:00 New York day before target; pre-publication training RT median',
        PUE_and_PV='held common between workload sources within each market year; 2024 rebuilt with dated NASA PV and frozen twin',
        DA_commitment='shared issue-time schedule from frozen pre-year empirical pools; original archived 2025 shared DA settlement component retained')
    save(BASE/'PROTOCOL.json',protocol)
    rows=[];quality=[]
    for year in [2024,2025]:
        folder=DATA/f'frontier_{year}';path=folder/f'frontier-{year}.swf'
        header=[]
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.startswith(';'):header.append(line.rstrip())
                else:break
        header='\n'.join(header)+'\n';(folder/'header.txt').write_text(header,encoding='utf-8')
        epoch=int(re.search(r'UnixStartTime:\s*(\d+)',header).group(1))
        declared_count=int(re.search(r'number of jobs:\s*(\d+)',header).group(1))
        f=pd.read_csv(path,sep=r'\s+',comment=';',header=None,names=FIELDS)
        properties=f.required_properties.str.extract(r'^frontier:excl:(\d+)x(\d+)$')
        assert properties.notna().all().all(),'Unrecognised resource property; no units may be guessed'
        f['requested_nodes']=properties[0].astype('int64');f['cores_per_node_property']=properties[1].astype('int64')
        year_start=pd.Timestamp(f'{year}-01-01',tz='UTC');year_end=pd.Timestamp(f'{year+1}-01-01',tz='UTC')
        duration=int((year_end-year_start).total_seconds());N=duration//900
        assert epoch==int(year_start.timestamp()) and len(f)==declared_count
        assert not f.job_id.duplicated().any() and f.submit_time.ge(0).all() and f.submit_time.lt(duration).all()
        assert f.requested_nodes.gt(0).all() and f.required_resource.gt(0).all()
        ids=(f.submit_time.to_numpy()//900).astype(int)
        counts=np.bincount(ids,minlength=N)
        nodes=np.bincount(ids,weights=f.requested_nodes,minlength=N)
        raw=np.bincount(ids,weights=f.required_resource,minlength=N)
        market_start=pd.Timestamp(f'{year}-01-01',tz='America/New_York').tz_convert('UTC')
        times=pd.date_range(market_start,periods=N,freq='15min')
        coeff=target/(.25*k.sum()/4*nodes.sum());rawcoeff=target/(.25*k.sum()/4*raw.sum())
        arrival=nodes[:,None]*coeff*np.ones((1,4))/4
        raw_arrival=raw[:,None]*rawcoeff*np.ones((1,4))/4
        assert abs(.25*np.sum(arrival*k)-target)<1e-8
        table=pd.DataFrame(dict(timestamp_utc=times,trace_elapsed_seconds=np.arange(N)*900,submitted_jobs=counts,requested_nodes=nodes,required_resource_field7=raw,total_work_units=nodes*coeff,total_work_units_field7=raw*rawcoeff))
        table.to_parquet(folder/'arrival_series.parquet',index=False)
        np.savez_compressed(folder/'mapped_arrivals.npz',timestamps_utc=times.tz_localize(None).to_numpy(),arrivals=arrival,resource_field7_arrivals=raw_arrival)
        f[['job_id','submit_time','requested_nodes','required_resource','allocated_resource','runtime','status','cores_per_node_property']].to_parquet(folder/'parsed_jobs.parquet',index=False)
        gaps=np.diff(np.sort(f.submit_time.to_numpy()));mean=float(gaps.mean());sd=float(gaps.std())
        daily=nodes.reshape(-1,96).sum(1)
        rec=dict(trace=f'Frontier-{year}',number_of_jobs=len(f),trace_start=pd.to_datetime(epoch+int(f.submit_time.min()),unit='s',utc=True),trace_end=pd.to_datetime(epoch+int(f.submit_time.max()),unit='s',utc=True),
            duration_days=float((f.submit_time.max()-f.submit_time.min())/86400),calendar_days=duration/86400,quarter_hours=N,
            mean_interarrival_seconds=mean,median_interarrival_seconds=float(np.median(gaps)),p95_interarrival_seconds=float(np.quantile(gaps,.95)),
            interarrival_burstiness=(sd-mean)/(sd+mean),interval_job_count_fano=float(counts.var()/counts.mean()),interval_requested_nodes_cv=float(nodes.std()/nodes.mean()),
            daily_workload_cv=float(daily.std()/daily.mean()),zero_arrival_interval_fraction=float((counts==0).mean()),
            peak_submitted_jobs_per_15min=int(counts.max()),peak_requested_nodes_per_15min=float(nodes.max()),peak_required_resource_field7_per_15min=float(raw.max()),
            epoch_zero_jobs=int(f.submit_time.eq(0).sum()),requested_node_sum=int(nodes.sum()),resource_field7_sum=int(raw.sum()),
            normalisation_coefficient_work_units_per_requested_node=coeff,field7_coefficient_work_units_per_resource=rawcoeff,
            annual_variable_it_mwh=float(.25*np.sum(arrival*k)),raw_sha256=sha(path),
            allocation_equals_required_field7=bool(f.allocated_resource.eq(f.required_resource).all()),field7_differs_from_requested_nodes_jobs=int(f.required_resource.ne(f.requested_nodes).sum()))
        for col in ['submit_time','requested_nodes','required_resource','runtime','allocated_resource']:
            values=f[col].to_numpy()
            for label,value in [('min',np.min(values)),('mean',np.mean(values)),('p50',np.quantile(values,.5)),('p95',np.quantile(values,.95)),('p99',np.quantile(values,.99)),('max',np.max(values))]:rec[f'{col}_{label}']=float(value)
        rows.append(rec)
        qr=dict(trace=rec['trace'],declared_count=declared_count,observed_count=len(f),duplicates=0,invalid_time=0,missing_requested_nodes=0,unparsed_properties=0,
            status_counts={str(k):int(v) for k,v in f.status.value_counts().items()},header_status_claim='always 1',header_status_claim_matches=False,
            excluded_jobs=0,excluded_days=0,capped_peaks=0,first_boundary_jobs_retained=rec['epoch_zero_jobs'],normalisation_mwh_residual=rec['annual_variable_it_mwh']-target)
        quality.append(qr);save(folder/'parse_quality.json',qr)
        (folder/'FIELDS.md').write_text('''# Header-grounded interpretation

The original header is retained in header.txt. Columns are zero-indexed.

- 0: unique job ID; 1: submission offset in seconds from UnixStartTime.
- 3: runtime in seconds, retained for diagnostics only.
- 4: allocated nodes or CPU-core resource count; 7: required count, equal to column 4 in this release.
- 20: required_properties. The header explicitly defines NxC as N required nodes, each with at least C cores. Every record matches frontier:excl:Nx112.

The primary arrival measure sums N from field 20 at submission. It follows the Eagle requested-node concept without treating a mixed node/core field as physical nodes. Field 7 is preserved and evaluated as a separately normalised sensitivity. Runtime, waiting time and job outcome do not weight the primary or sensitivity demand. All statuses, epoch-zero jobs, large requests and zero-arrival intervals remain in the replay. The header's statement that status is always 1 does not match observed records; status_counts are retained in parse_quality.json.

The counterfactual calendar starts at NYISO local Jan 1 midnight and uses unchanged trace elapsed seconds. This is a constant five-hour UTC phase shift, not a claim of contemporaneous Frontier operation in New York. It preserves full 366/365-day duration and all intra-trace event spacing. Annual normalisation is fixed to the exact frozen Eagle-2025 variable IT work before economic evaluation.
''',encoding='utf-8')
        print(json.dumps({k:rec[k] for k in ['trace','number_of_jobs','calendar_days','annual_variable_it_mwh','normalisation_coefficient_work_units_per_requested_node','peak_requested_nodes_per_15min','daily_workload_cv','zero_arrival_interval_fraction']},default=str),flush=True)
    pd.DataFrame(rows).to_csv(RESULTS/'frontier_trace_characteristics.csv',index=False)
    save(DATA/'frontier_parse_quality.json',quality)
    frozen=[PAPER/'revision_20260918/annual_continuous_2025/data/annual_forcing_2025.npz',PAPER/'revision_20260920_r2_executable/results/policy_comparison_2025.csv',PAPER/'revision_20260920_r2_executable/results/heterogeneous_deadline_R2_results.csv',PAPER/'revision_20260922_completion/results/traditional_baseline_economics_12h.csv',PAPER/'交付稿_20260922_完整修订与中文对照/NatureEnergy_ExecutableFlexibility_20260922_EN.docx']
    save(BASE/'frozen_2025_manifest.json',[dict(path=str(p.resolve()),sha256=sha(p)) for p in frozen])
if __name__=='__main__':main()
