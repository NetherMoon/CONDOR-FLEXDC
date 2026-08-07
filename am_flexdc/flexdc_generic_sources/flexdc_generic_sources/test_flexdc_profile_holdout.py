#!/usr/bin/env python3
"""Structural tests for the four strict J=2 profile-holdout splits."""
from __future__ import annotations
import itertools, json
import pandas as pd
from flexdc_profile_holdout import MODEL_SPECS, assign_profile_holdout_split

families=['ResNet','GPT-2','Llama','Bloom']
rows=[]
workloads=[]
for mode,code,suffix in [('Inference','II','Inf'),('Training','TT','Train')]:
 for a,b in itertools.combinations(families,2):
  workloads.append((f"J2-{code}-{a.replace('-','')}{suffix}-{b.replace('-','')}{suffix}", [f'{a} {mode}',f'{b} {mode}']))
for inf in families:
 for train in families:
  workloads.append((f"J2-IT-{inf.replace('-','')}Inf-{train.replace('-','')}Train",[f'{inf} Inference',f'{train} Training']))
assert len(workloads)==28
for wname,profiles in workloads:
 for context_u in [0.6,0.8]:
  for group in range(20):
   for seed in [20,21] if group==0 else [20]:
    rows.append({'Plan_Row_ID':f'{wname}_{context_u}_{group}_{seed}','Base_Plan_Row_ID':f'{wname}_{context_u}_{group}','Workload_Name':wname,'workload_config':f'configs/workload/j2_pairwise/{wname}.ini','server_count':1000,'utilization':context_u,'simulation_seed':seed,'Job_Profiles':json.dumps(profiles)})
master=pd.DataFrame(rows)
expected={'3x1_llama_holdout':15,'3x1_resnet_holdout':15,'2x2_train_resnet_gpt2':6,'2x2_train_llama_bloom':6}
for model_id in MODEL_SPECS:
 split,audit=assign_profile_holdout_split(master,model_id,split_seed=7,expected_jobs=2,strict_master_coverage=True)
 assert audit['status']=='PASS'
 assert audit['workload_counts']['seen_profile_workloads']==expected[model_id]
 assert set(split['Data_Split'])=={'train','validation','test'}
 assert not ((split['Heldout_Profile_Count']>0)&(split['Data_Split']!='test')).any()
 assert (split.groupby('Base_Plan_Row_ID')['Data_Split'].nunique()==1).all()
 assert set(split.loc[split['Heldout_Profile_Count']==1,'Benchmark_Type'])=={'One Held-Out Profile'}
 assert set(split.loc[split['Heldout_Profile_Count']==2,'Benchmark_Type'])=={'Two Held-Out Profiles'}
 print(model_id, 'PASS', audit['workload_counts'])
print('All profile-holdout structural tests passed.')
