"""Reproduce the discussed numeric example with full precision; no VLM calls."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from .math_core import history_and_risk, discounted_cost_to_go, leave_one_out, ppo_clip_loss, jsd_from_logits


def generate() -> dict:
    dt=torch.float64
    cost=torch.tensor([[.2,.3,.3,.1,.3,.2],[.2,.3,.3,.1,.02,0]],dtype=dt)
    mask=torch.ones_like(cost,dtype=torch.bool)
    risk=history_and_risk(cost,mask,'bounded_slope')
    ret=discounted_cost_to_go(risk['increment'],mask,1.0)
    feedback=leave_one_out(ret,mask,torch.tensor([0,0]))
    t=3
    a=feedback['advantage'][0:1,t:t+1]
    ppo=ppo_clip_loss(torch.tensor([[.63]],dtype=dt).log(),
                      torch.tensor([[.70]],dtype=dt).log(),a,torch.ones((1,1),dtype=torch.bool),.2)
    jsd=jsd_from_logits(torch.tensor([[[.63,.27,.1]]],dtype=dt).log(),
                         torch.tensor([[[.7,.2,.1]]],dtype=dt).log(),torch.ones((1,1),dtype=torch.bool))
    return {
        'note':'Synthetic illustration only; no model training, no performance result.',
        'indexing':'current_index=3 is response position 4 in 1-based formulas',
        'cost_kind':'probability_gap','gamma':1.0,'clip_epsilon_demo':.2,'lambda_demo':.1,
        'cost':cost.tolist(),'history':risk['history'].tolist(),
        'risk_increment':risk['increment'].tolist(),'returns':ret.tolist(),
        'baseline':feedback['baseline'].tolist(),'advantage':feedback['advantage'].tolist(),
        'current_index':t,'current_returns':ret[:,t].tolist(),
        'current_advantages':feedback['advantage'][:,t].tolist(),
        'actor_example':{'old_probability':.7,'new_probability':.63,
             'ratio':ppo['ratio'].item(),'policy_loss':ppo['loss'].item(),
             'anchor_jsd':jsd.item(),'total_loss':(ppo['loss']+.1*jsd).item()},
    }

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    data=generate()
    dest=root/'examples'/'worked_example.json'
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data['actor_example'],indent=2))
    print(f'Wrote {dest}')
