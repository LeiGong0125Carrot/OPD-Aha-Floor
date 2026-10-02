from pathlib import Path
import copy
import yaml
from reference.check_config import validate_config
from reference.generate_example import generate
import pytest

ROOT=Path(__file__).resolve().parents[1]

def test_template_intentionally_blocks_unlocked_run():
    cfg=yaml.safe_load((ROOT/'configs/train_plan.yaml').read_text())
    issues=validate_config(cfg)
    assert 'missing: feedback.gamma' in issues
    assert 'missing: anchor.enabled' in issues


def test_all_decisions_filled_and_branch_constraints_checked():
    cfg=yaml.safe_load((ROOT/'configs/train_plan.yaml').read_text())
    for key in cfg['inherit']:
        cfg['inherit'][key]= 'explicit_test_manifest'
    cfg['inherit']['rollout_count_per_prompt']=2
    cfg['feedback']['teacher_scoring_temperature']=1
    cfg['feedback']['gamma']=.9
    cfg['optimization']['clip_epsilon']=.2
    cfg['optimization']['ppo_epochs']=1
    cfg['anchor']['enabled']=False
    cfg['anchor']['weight']=0
    assert validate_config(cfg)==[]
    bad=copy.deepcopy(cfg)
    bad['constraints']['add_gu_to_ppo']=True
    assert any('GU' in s for s in validate_config(bad))


def test_full_example_numeric_reproducible():
    example=generate()
    assert example['current_returns'][0]==pytest.approx(.912317927548219)
    assert example['current_returns'][1]==pytest.approx(.175461478862429)
    assert example['actor_example']['total_loss']==pytest.approx(.663524562180332)
