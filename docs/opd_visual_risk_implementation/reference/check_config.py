"""Validate explicit decisions. Does not resolve manifests or start training."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml


def validate_config(cfg: dict) -> list[str]:
    issues: list[str] = []
    def need(section: str, name: str):
        value = cfg.get(section, {}).get(name)
        if value is None:
            issues.append(f'missing: {section}.{name}')
        return value
    mode=cfg.get('mode')
    if mode not in ('temporal_visual_risk_ppo','gu_local'):
        issues.append('mode must select one objective branch')
    if cfg.get('seed') != 42:
        issues.append('seed must preserve the agreed seed=42 constraint')
    for name in ('repository_root','baseline_manifest','student_checkpoint',
                 'frozen_teacher_checkpoint','train_data_manifest','processor_manifest',
                 'sampling_spec','optimizer_manifest','update_budget','evaluation_manifest'):
        need('inherit',name)
    n=need('inherit','rollout_count_per_prompt')
    if n is not None and (not isinstance(n,int) or n<1):
        issues.append('rollout_count_per_prompt must be a positive integer')
    if mode=='temporal_visual_risk_ppo' and isinstance(n,int) and n<2:
        issues.append('LOO requires n>=2; do not silently increase the rollout budget')
    tau=need('feedback','teacher_scoring_temperature')
    if tau is not None and (not isinstance(tau,(int,float)) or tau<=0):
        issues.append('teacher scoring temperature must be positive')
    f=cfg.get('feedback',{})
    if f.get('cost_kind') not in ('probability_gap','negative_log_ratio'):
        issues.append('unknown cost_kind')
    if mode=='temporal_visual_risk_ppo':
        gamma=need('feedback','gamma')
        if gamma is not None and (not isinstance(gamma,(float,int)) or not 0<=gamma<=1):
            issues.append('gamma must be in [0,1]')
        eps=need('optimization','clip_epsilon')
        if eps is not None and (not isinstance(eps,(float,int)) or not 0<eps<1):
            issues.append('clip epsilon must be in (0,1)')
        epochs=need('optimization','ppo_epochs')
        if epochs is not None and (not isinstance(epochs,int) or epochs<1):
            issues.append('ppo_epochs must be a positive integer')
    o=cfg.get('optimization',{})
    if o.get('loss_reduction') not in ('token_mean','trajectory_sum_mean'):
        issues.append('loss reduction must be explicit')
    if o.get('advantage_normalization')!='none':
        issues.append('reference implementation does not silently whiten advantages')
    enabled=need('anchor','enabled')
    weight=need('anchor','weight')
    if enabled is not None and not isinstance(enabled,bool):
        issues.append('anchor.enabled must be a boolean')
    if weight is not None and (not isinstance(weight,(int,float)) or weight<0):
        issues.append('anchor weight must be nonnegative')
    if enabled is False and weight not in (None,0):
        issues.append('disabled anchor requires weight=0')
    if enabled is True and weight==0:
        issues.append('enabled anchor with zero weight is ambiguous; disable explicitly')
    c=cfg.get('constraints',{})
    if c.get('add_gu_to_ppo'):
        issues.append('GU must remain a separate objective branch')
    if c.get('teacher_views')!=['privileged','matched_mean_rgb_null']:
        issues.append('teacher views must remain the two agreed views')
    for key in ('add_critic','add_attention_module','standalone_scientific_probes','multi_seed_sweep'):
        if c.get(key):
            issues.append(f'forbidden change: {key}')
    if c.get('extra_lookahead_rollouts',0)!=0:
        issues.append('do not add lookahead sampling')
    return issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args=parser.parse_args()
    cfg=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(cfg,dict):
        raise SystemExit('config must be a mapping')
    issues=validate_config(cfg)
    if issues:
        print('NOT READY TO LAUNCH:')
        for issue in issues:
            print(' -', issue)
        raise SystemExit(2)
    print('Explicit configuration checks passed. Repository/model adapters are still required.')

if __name__=='__main__':
    main()
