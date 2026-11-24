import torch
import torch.nn.functional as F

from src.models.compat import CompatModel, PairScorer, batch_hard_triplet_loss, triplet_loss


def reference_forward(m: CompatModel, x, cat):
    """Per-row loop: each row through its own category's 2-layer MLP."""
    out = []
    for xi, c in zip(x, cat):
        h = F.relu(xi @ m.w1[c] + m.b1[c])
        out.append(h @ m.w2[c] + m.b2[c])
    return F.normalize(torch.stack(out), dim=-1)


def test_compat_model_matches_per_row_reference():
    torch.manual_seed(0)
    m = CompatModel(n_categories=5, d_in=24, d_hidden=16, d_out=8)
    x, cat = torch.randn(40, 24), torch.randint(0, 5, (40,))
    torch.testing.assert_close(m(x, cat), reference_forward(m, x, cat), rtol=1e-5, atol=1e-5)
    assert torch.allclose(m(x, cat).norm(dim=-1), torch.ones(40))


def test_gradients_only_reach_used_categories():
    m = CompatModel(n_categories=4, d_in=8, d_hidden=6, d_out=4)
    m(torch.randn(10, 8), torch.tensor([0, 2] * 5)).sum().backward()
    g = m.w1.grad.abs().sum(dim=(1, 2))
    assert g[0] > 0 and g[2] > 0 and g[1] == 0 and g[3] == 0


def test_triplet_losses():
    a = F.normalize(torch.randn(6, 4), dim=-1)
    assert triplet_loss(a, a, -a, margin=0.3) == 0          # perfect separation
    assert triplet_loss(a, -a, a, margin=0.3) > 0
    cat = torch.zeros(6, dtype=torch.long)
    outfit = torch.arange(6)
    # in-batch negatives can only make the loss harder (>=), never easier
    assert batch_hard_triplet_loss(a, a, -a, cat, outfit, 0.3) >= triplet_loss(a, a, -a, 0.3)
    # if every other positive is the same item, nothing in-batch is a valid negative
    same = torch.zeros(6, dtype=torch.long)
    torch.testing.assert_close(batch_hard_triplet_loss(a, a, -a, cat, outfit, 0.3, pos_item=same),
                               triplet_loss(a, a, -a, 0.3))


def test_pair_scorer_symmetric():
    torch.manual_seed(0)
    s = PairScorer(d=8, n_color=0).eval()
    a, b = torch.randn(5, 8), torch.randn(5, 8)
    torch.testing.assert_close(s.symmetric(a, b), s.symmetric(b, a))
