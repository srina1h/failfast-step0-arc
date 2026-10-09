import torch

from failfast_arc.model import compute_loss


def test_pairwise_loss_orders_pairs():
    labels = torch.tensor([1.0, 0.0])
    good, good_parts = compute_loss(torch.tensor([2.0, -2.0], requires_grad=True), labels, torch.tensor([0]), torch.tensor([1]), 0.25)
    bad, bad_parts = compute_loss(torch.tensor([-2.0, 2.0], requires_grad=True), labels, torch.tensor([0]), torch.tensor([1]), 0.25)
    assert torch.isfinite(good)
    assert good_parts["rank"] < bad_parts["rank"]
    good.backward()


def test_empty_pairs_have_zero_rank_loss():
    loss, parts = compute_loss(torch.tensor([0.0], requires_grad=True), torch.tensor([1.0]), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), 0.25)
    assert parts["rank"].item() == 0.0
    assert torch.isfinite(loss)
