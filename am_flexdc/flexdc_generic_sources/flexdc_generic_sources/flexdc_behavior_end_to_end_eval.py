#!/usr/bin/env python3
"""Optimize one workload with one checkpoint and optionally validate top-k in FlexDC."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import pandas as pd
from flexdc_behavior_inference_utilities import (
    OptimizationSettings, calculate_pr_bounds, dataframe_for_csv, load_behavior_model,
    optimize_candidates, read_experiment_config, read_workload_config,
    resolve_effective_weight_bounds, resolve_safety_limits, run_flexdc_validation,
    write_json,
)
def parse_weights(text): return None if text is None else [float(x.strip()) for x in text.split(',') if x.strip()]
def parse_json_list(value):
    if isinstance(value,list): return [float(x) for x in value]
    return [float(x) for x in json.loads(str(value))]
def parser():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True); p.add_argument('--workload-config',required=True); p.add_argument('--experiment-config',required=True); p.add_argument('--server-count',type=int); p.add_argument('--utilization',type=float); p.add_argument('--device',default='auto')
 p.add_argument('--initial-pbar',type=float); p.add_argument('--initial-r',type=float); p.add_argument('--initial-weights')
 p.add_argument('--mode',choices=['pure_objective','exact_constrained','margin_constrained'],default='margin_constrained'); p.add_argument('--tracking-limit',type=float); p.add_argument('--qos-limit',type=float); p.add_argument('--tracking-margin',type=float,default=0.04); p.add_argument('--qos-margin',type=float,default=0.01)
 p.add_argument('--starts',type=int,default=512); p.add_argument('--iterations',type=int,default=1500); p.add_argument('--learning-rate',type=float,default=0.03); p.add_argument('--minimum-learning-rate',type=float,default=5e-4); p.add_argument('--tracking-penalty',type=float,default=2000); p.add_argument('--qos-penalty',type=float,default=2000); p.add_argument('--penalty-ramp-fraction',type=float,default=0.30); p.add_argument('--top-k',type=int,default=5); p.add_argument('--candidate-distance',type=float,default=0.03); p.add_argument('--random-seed',type=int,default=0); p.add_argument('--near-equal-start-fraction',type=float,default=0.25); p.add_argument('--high-p-low-r-start-fraction',type=float,default=0.25); p.add_argument('--log-every',type=int,default=25)
 p.add_argument('--pbar-lower-factor',type=float,default=0.9); p.add_argument('--pbar-upper-factor',type=float,default=1.0); p.add_argument('--pr-upper-factor',type=float,default=1.2); p.add_argument('--r-lower',type=float,default=0.01); p.add_argument('--r-over-p-max',type=float,default=0.6); p.add_argument('--pbar-min',type=float); p.add_argument('--pbar-max',type=float); p.add_argument('--r-min',type=float); p.add_argument('--r-max',type=float)
 p.add_argument('--enforce-flexdc-weight-bounds',action=argparse.BooleanOptionalAction,default=True); p.add_argument('--weight-min-fraction-of-equal',type=float,default=0.1); p.add_argument('--weight-max-multiple-of-equal',type=float,default=4.0); p.add_argument('--weight-min',type=float); p.add_argument('--weight-max',type=float)
 p.add_argument('--run-flexdc-validation',action=argparse.BooleanOptionalAction,default=False); p.add_argument('--flexdc-root'); p.add_argument('--gradient-config'); p.add_argument('--cluster-config'); p.add_argument('--policy-name',default='AQA'); p.add_argument('--python-executable',default=sys.executable); p.add_argument('--validation-timeout',type=int,default=1800); p.add_argument('--dry-run-flexdc',action='store_true')
 p.add_argument('--out-dir',required=True); p.add_argument('--run-name',default='behavior_model_e2e'); return p
def main():
 a=parser().parse_args(); out=Path(a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
 loaded=load_behavior_model(a.checkpoint,device_name=a.device); workload=read_workload_config(a.workload_config); experiment=read_experiment_config(a.experiment_config,server_count_override=a.server_count,utilization_override=a.utilization)
 bounds=calculate_pr_bounds(workload,pbar_lower_factor=a.pbar_lower_factor,pbar_upper_factor=a.pbar_upper_factor,pr_upper_factor=a.pr_upper_factor,r_lower_kw_per_server=a.r_lower); safety=resolve_safety_limits(loaded.constants,tracking_limit=a.tracking_limit,qos_limit=a.qos_limit,tracking_margin=a.tracking_margin,qos_margin=a.qos_margin)
 settings=OptimizationSettings(starts=a.starts,iterations=a.iterations,learning_rate=a.learning_rate,minimum_learning_rate=a.minimum_learning_rate,mode=a.mode,tracking_penalty=a.tracking_penalty,qos_penalty=a.qos_penalty,penalty_ramp_fraction=a.penalty_ramp_fraction,top_k=a.top_k,candidate_distance=a.candidate_distance,random_seed=a.random_seed,near_equal_start_fraction=a.near_equal_start_fraction,high_p_low_r_start_fraction=a.high_p_low_r_start_fraction,enforce_flexdc_weight_bounds=a.enforce_flexdc_weight_bounds,weight_min_fraction_of_equal=a.weight_min_fraction_of_equal,weight_max_multiple_of_equal=a.weight_max_multiple_of_equal,weight_min=a.weight_min,weight_max=a.weight_max,r_over_p_max=a.r_over_p_max,pbar_min_override=a.pbar_min,pbar_max_override=a.pbar_max,r_min_override=a.r_min,r_max_override=a.r_max,log_every=a.log_every)
 effective=resolve_effective_weight_bounds(settings,job_count=workload.job_count,server_count=experiment.server_count); candidates,top_k,trajectory=optimize_candidates(loaded,workload=workload,experiment=experiment,bounds=bounds,safety=safety,settings=settings,initial_pbar=a.initial_pbar,initial_reserve=a.initial_r,initial_weights=parse_weights(a.initial_weights))
 dataframe_for_csv(candidates).to_csv(out/f'{a.run_name}_all_starts.csv',index=False); dataframe_for_csv(top_k).to_csv(out/f'{a.run_name}_top_k_predicted.csv',index=False); dataframe_for_csv(trajectory).to_csv(out/f'{a.run_name}_trajectory.csv',index=False)
 validations=[]; per_jobs=[]
 if a.run_flexdc_validation:
  missing=[name for name in ['flexdc_root','gradient_config','cluster_config'] if getattr(a,name) in [None,'']]
  if missing: raise ValueError('--run-flexdc-validation requires: '+', '.join('--'+x.replace('_','-') for x in missing))
  for _,row in top_k.iterrows():
   rank=int(row['Candidate_Rank']); weights=row['weights'] if isinstance(row['weights'],list) else parse_json_list(row['weights']); label=re.sub(r'[^A-Za-z0-9_.-]+','_',f'{a.run_name}_rank_{rank}')
   actual,jobs=run_flexdc_validation(python_executable=a.python_executable,flexdc_root=a.flexdc_root,gradient_config=a.gradient_config,experiment_config=a.experiment_config,cluster_config=a.cluster_config,workload_config=a.workload_config,output_label=label,pbar_kw_per_server=float(row['Pbar_kw_per_server']),r_kw_per_server=float(row['R_kw_per_server']),weights=weights,utilization=experiment.utilization,constants=loaded.constants,policy_name=a.policy_name,node_count_control=True,timeout_seconds=a.validation_timeout,dry_run=a.dry_run_flexdc)
   record={'Candidate_Rank':rank,'Pbar_kw_per_server':float(row['Pbar_kw_per_server']),'R_kw_per_server':float(row['R_kw_per_server']),'weights':weights,**{k:v for k,v in row.to_dict().items() if str(k).startswith('Predicted_') or str(k).endswith('_Pass') or str(k).endswith('_Slack')},**actual}; validations.append(record)
   if not a.dry_run_flexdc and len(jobs): jobs=jobs.copy(); jobs.insert(0,'Candidate_Rank',rank); jobs['Job_Type']=workload.job_names; per_jobs.append(jobs)
 validation=pd.DataFrame(validations); dataframe_for_csv(validation).to_csv(out/f'{a.run_name}_predicted_vs_actual.csv',index=False)
 if per_jobs: dataframe_for_csv(pd.concat(per_jobs,ignore_index=True)).to_csv(out/f'{a.run_name}_per_job_qos.csv',index=False)
 selected=None; status='surrogate_optimization_complete'
 if a.run_flexdc_validation and not a.dry_run_flexdc:
  if len(validation) and 'Actual_Both_Pass' in validation.columns:
   feasible = validation[validation['Actual_Both_Pass'].astype(bool)]
  else:
   feasible = pd.DataFrame()
  if len(feasible): selected=feasible.sort_values('Actual_Full_Objective').iloc[0].to_dict(); status='simulator_feasible_candidate_selected'
  else: status='no_simulator_feasible_candidate'
 summary={'status':status,'checkpoint':str(Path(a.checkpoint).resolve()),'checkpoint_epoch':int(loaded.checkpoint.get('epoch',-1)),'workload_config':str(Path(a.workload_config).resolve()),'experiment_config':str(Path(a.experiment_config).resolve()),'settings':settings.__dict__,'safety':safety.to_dict(),'effective_weight_bounds':effective.to_dict(),'exact_feasible_starts':int(candidates['Exact_Both_Pass'].sum()),'safety_feasible_starts':int(candidates['Safety_Both_Pass'].sum()),'top_k_count':int(len(top_k)),'flexdc_validations':int(len(validation)),'selected_actual_feasible_candidate':selected}; write_json(out/f'{a.run_name}_summary.json',summary); print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
