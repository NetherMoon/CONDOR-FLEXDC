#!/usr/bin/env python3
"""Predict one FlexDC configuration with one generic behavior-model checkpoint."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from flexdc_behavior_inference_utilities import (
    calculate_pr_bounds, dataframe_for_csv, load_behavior_model, predict_configuration,
    read_experiment_config, read_workload_config, resolve_safety_limits, write_json,
)

def weights(text: str): return [float(x.strip()) for x in text.split(',') if x.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--workload-config',required=True); p.add_argument('--experiment-config',required=True)
    p.add_argument('--pbar',type=float,required=True); p.add_argument('--r',type=float,required=True); p.add_argument('--weights',required=True)
    p.add_argument('--server-count',type=int); p.add_argument('--utilization',type=float); p.add_argument('--device',default='auto')
    p.add_argument('--tracking-limit',type=float); p.add_argument('--qos-limit',type=float); p.add_argument('--tracking-margin',type=float,default=0.04); p.add_argument('--qos-margin',type=float,default=0.01)
    p.add_argument('--pbar-lower-factor',type=float,default=0.9); p.add_argument('--pbar-upper-factor',type=float,default=1.0); p.add_argument('--pr-upper-factor',type=float,default=1.2); p.add_argument('--r-lower',type=float,default=0.01); p.add_argument('--r-over-p-max',type=float)
    p.add_argument('--out-dir',default='.'); p.add_argument('--run-name',default='predict_one')
    a=p.parse_args(); out=Path(a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    loaded=load_behavior_model(a.checkpoint,device_name=a.device); workload=read_workload_config(a.workload_config)
    experiment=read_experiment_config(a.experiment_config,server_count_override=a.server_count,utilization_override=a.utilization)
    bounds=calculate_pr_bounds(workload,pbar_lower_factor=a.pbar_lower_factor,pbar_upper_factor=a.pbar_upper_factor,pr_upper_factor=a.pr_upper_factor,r_lower_kw_per_server=a.r_lower)
    safety=resolve_safety_limits(loaded.constants,tracking_limit=a.tracking_limit,qos_limit=a.qos_limit,tracking_margin=a.tracking_margin,qos_margin=a.qos_margin)
    result, jobs=predict_configuration(loaded,workload=workload,experiment=experiment,pbar_kw_per_server=a.pbar,r_kw_per_server=a.r,weights=weights(a.weights),safety=safety,bounds=bounds,r_over_p_max=a.r_over_p_max)
    write_json(out/f'{a.run_name}_prediction.json',result); dataframe_for_csv(jobs).to_csv(out/f'{a.run_name}_per_job.csv',index=False)
    print(json.dumps(result,indent=2)); print('\nPer-job QoS\n',jobs.to_string(index=False))
if __name__=='__main__': main()
