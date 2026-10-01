import math
import pytest
import torch
from reference.math_core import (visual_cost, history_and_risk, discounted_cost_to_go,
    leave_one_out, ppo_clip_loss, jsd_from_logits, gu_loss, reduce_positions)

DT = torch.float64

def tensor(x):
    return torch.tensor(x, dtype=DT)

def fullmask(shape):
    return torch.ones(shape, dtype=torch.bool)


def test_gap_log_and_sign():
    p = tensor([[.6, .3, .1]])
    n = tensor([[.8, .1, .1]])
    m = fullmask(p.shape)
    out = visual_cost(p.log(), n.log(), m)
    torch.testing.assert_close(out['cost'], tensor([[.2, 0, 0]]))
    out_log = visual_cost(p.log(), n.log(), m, 'negative_log_ratio')
    torch.testing.assert_close(out_log['cost'], tensor([[math.log(4/3), 0, 0]]))


def test_mask_removes_padding_nan():
    lp = tensor([[math.log(.6), float('nan')]])
    ln = tensor([[math.log(.8), float('nan')]])
    m = torch.tensor([[True, False]])
    out = visual_cost(lp, ln, m)
    torch.testing.assert_close(out['cost'], tensor([[.2, 0]]))


def test_history_exclusive_and_telescoping():
    c = tensor([[.2, .3, .3, .1, .3, .2]])
    m = fullmask(c.shape)
    out = history_and_risk(c, m)
    torch.testing.assert_close(out['history'], tensor([[0, .2, .5, .8, .9, 1.2]]))
    r = discounted_cost_to_go(out['increment'], m, 1)
    phi = lambda x: 2*x-math.log1p(x)
    assert r[0,3].item() == pytest.approx(phi(1.4)-phi(.8))
    assert out['increment'].sum().item() == pytest.approx(phi(1.4))


def test_bounded_risk_increment_and_zero():
    c = tensor([[2, .2, 0]])
    m = fullmask(c.shape)
    out = history_and_risk(c, m)
    d = out['increment']
    assert torch.all(d >= c - 1e-12)
    assert torch.all(d <= 2*c + 1e-12)
    assert d[0,2].item() == 0
    assert d[0,1].item() == pytest.approx(.4-math.log(3.2/3))


def test_linear_and_gamma_endpoints():
    c = tensor([[.1,.3,0,.2]])
    m = fullmask(c.shape)
    d = history_and_risk(c,m,'linear')['increment']
    torch.testing.assert_close(d,c)
    torch.testing.assert_close(discounted_cost_to_go(d,m,0),c)
    torch.testing.assert_close(discounted_cost_to_go(d,m,1),tensor([[.6,.5,.2,.2]]))
    assert discounted_cost_to_go(d,m,.9)[0,0].item() == pytest.approx(.5158)


def test_padding_absorbing_returns():
    d = tensor([[.1,.2,999],[.3,999,999]])
    m = torch.tensor([[True,True,False],[True,False,False]])
    torch.testing.assert_close(discounted_cost_to_go(d,m,1),tensor([[.3,.2,0],[.3,0,0]]))


def test_loo_two_sequences_and_isolation():
    r = tensor([[.6,.5],[.12,.02],[10,5],[20,7]])
    m = fullmask(r.shape)
    out = leave_one_out(r,m,torch.tensor([0,0,1,1]))
    torch.testing.assert_close(out['advantage'][:2],tensor([[-.48,-.48],[.48,.48]]))
    torch.testing.assert_close(out['advantage'][2:],tensor([[10,2],[-10,-2]]))


def test_loo_terminated_peer_kept_in_denominator():
    r = tensor([[.6,.5],[.12,0]])
    m = torch.tensor([[True,True],[True,False]])
    out = leave_one_out(r,m,torch.tensor([0,0]))
    assert out['baseline'][0,1].item() == 0
    assert out['advantage'][0,1].item() == -.5
    assert out['advantage'][1,1].item() == 0


def test_loo_singleton_is_error():
    with pytest.raises(ValueError, match='>=2'):
        leave_one_out(tensor([[.1]]), fullmask((1,1)), torch.tensor([0]))


@pytest.mark.parametrize('pnew,adv,expected',[
    (.7,-.737,.737),(.63,-.737,.6633),(.56,-.737,.5896),(.35,-.737,.5896),
])
def test_ppo_negative_example(pnew,adv,expected):
    out = ppo_clip_loss(tensor([[math.log(pnew)]]),tensor([[math.log(.7)]]),
                        tensor([[adv]]),fullmask((1,1)),.2)
    assert out['loss'].item() == pytest.approx(expected)


@pytest.mark.parametrize('pnew,expected',[(.1,-.737),(.11,-.8107),(.12,-.8844),(.2,-.8844)])
def test_ppo_positive_example(pnew,expected):
    out = ppo_clip_loss(tensor([[math.log(pnew)]]),tensor([[math.log(.1)]]),
                        tensor([[.737]]),fullmask((1,1)),.2)
    assert out['loss'].item() == pytest.approx(expected)


def test_ppo_clipped_own_gradient_zero():
    new = tensor([[math.log(.35)]]).requires_grad_()
    out = ppo_clip_loss(new,tensor([[math.log(.7)]]),tensor([[-.737]]),fullmask((1,1)),.2)
    out['loss'].backward()
    assert new.grad.item() == 0


def test_fixed_teacher_feedback_has_no_grad():
    lp = tensor([[math.log(.6)]]).requires_grad_()
    ln = tensor([[math.log(.8)]]).requires_grad_()
    out = visual_cost(lp,ln,fullmask((1,1)))
    assert not out['cost'].requires_grad
    d = history_and_risk(out['cost'],fullmask((1,1)))['increment']
    assert not d.requires_grad


def test_ppo_detaches_old_and_advantage():
    new = tensor([[math.log(.63)]]).requires_grad_()
    old = tensor([[math.log(.7)]]).requires_grad_()
    adv = tensor([[-.737]]).requires_grad_()
    out = ppo_clip_loss(new,old,adv,fullmask((1,1)),.2)
    out['loss'].backward()
    assert new.grad is not None and new.grad.item()>0
    assert old.grad is None and adv.grad is None


def test_jsd_numeric_zero_and_gradient():
    t = tensor([[[.6,.3,.1]]]).log().requires_grad_()
    s = tensor([[[.3,.6,.1]]]).log().requires_grad_()
    m = fullmask((1,1))
    jsd = jsd_from_logits(s,t,m)
    assert jsd.item() == pytest.approx(.05096971103661828)
    jsd.sum().backward()
    assert s.grad is not None and s.grad.abs().sum()>0
    assert t.grad is None
    assert jsd_from_logits(t.detach(),t.detach(),m).item() == pytest.approx(0,abs=1e-12)


def test_jsd_example_three_probs():
    s = tensor([[[.63,.27,.1]]]).log()
    t = tensor([[[.7,.2,.1]]]).log()
    v = jsd_from_logits(s,t,fullmask((1,1))).item()
    assert v == pytest.approx(.0035375836312127144,rel=1e-8)


def test_gu_gate_zero_and_probability_direction():
    m = fullmask((1,1))
    hi = gu_loss(tensor([[math.log(.7)]]),tensor([[.2]]),m)
    lo = gu_loss(tensor([[math.log(.4)]]),tensor([[.2]]),m)
    assert hi.item() == pytest.approx(.2/1.3)
    assert lo.item() < hi.item()
    z = tensor([[math.log(.7)]]).requires_grad_()
    g = tensor([[0.]]).requires_grad_()
    loss = gu_loss(z,g,m)
    loss.backward()
    assert z.grad.item() == 0 and g.grad is None


def test_sequence_ratio_identity():
    plus = tensor([.3,.4,.6])
    null = tensor([.6,.5,.5])
    assert (plus.log()-null.log()).sum().item() == pytest.approx(math.log(.48))


def test_reductions_are_not_silently_equal():
    x = tensor([[1.,1.],[2.,0.]])
    m = torch.tensor([[True,True],[True,False]])
    assert reduce_positions(x,m,'token_mean').item()==pytest.approx(4/3)
    assert reduce_positions(x,m,'trajectory_sum_mean').item()==pytest.approx(2)


def test_history_resets_per_response():
    c=tensor([[.5,.2],[.1,.1]])
    h=history_and_risk(c,fullmask(c.shape))['history']
    assert h[0,0].item()==0 and h[1,0].item()==0
    assert h[1,1].item()==pytest.approx(.1)


def test_wrong_mask_and_invalid_logp_fail():
    with pytest.raises(ValueError,match='hole'):
        visual_cost(tensor([[-1,-1,-1]]),tensor([[-1,-1,-1]]),torch.tensor([[True,False,True]]))
    with pytest.raises(ValueError,match='<= 0'):
        visual_cost(tensor([[.1]]),tensor([[-1.]]),fullmask((1,1)))


def test_no_visual_signal_no_policy_advantage():
    lp=tensor([[-1.,-2.],[-1.5,-2.5]])
    m=fullmask(lp.shape)
    c=visual_cost(lp,lp,m)['cost']
    d=history_and_risk(c,m)['increment']
    r=discounted_cost_to_go(d,m,.9)
    a=leave_one_out(r,m,torch.tensor([0,0]))['advantage']
    assert a.abs().sum().item()==0


def test_terminal_eos_position_included_by_mask():
    # The last True position stands for an actually emitted EOS, not padding.
    c=tensor([[.1,.2,0.]])
    m=torch.tensor([[True,True,False]])
    r=discounted_cost_to_go(c,m,1)
    assert r[0,0].item()==pytest.approx(.3)
    assert r[0,1].item()==pytest.approx(.2)
