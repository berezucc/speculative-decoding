import torch
import torch.nn.functional as F
from acceptance import acceptance_prob, sample_residual


def test_p_equals_q_always_accepts():
    logits = torch.randn(1000)
    p = F.softmax(logits, dim=-1)
    q = p.clone()
    for token_id in range(len(p)):
        prob = acceptance_prob(p, q, token_id)
        assert abs(prob - 1.0) < 1e-5, f"expected 1.0, got {prob}"
    print("PASS: p == q, always accept")


def test_p_dominates_always_accepts():
    p = torch.zeros(100); p[42] = 1.0
    q = torch.zeros(100); q[42] = 0.01; q[0] = 0.99
    assert acceptance_prob(p, q, 42) == 1.0
    print("PASS: p >> q, always accept")


def test_q_dominates_often_rejects():
    p = torch.zeros(100); p[0] = 0.01; p[1] = 0.99
    q = torch.zeros(100); q[0] = 1.0
    prob = acceptance_prob(p, q, 0)
    assert prob < 0.02, f"expected near 0, got {prob}"
    print(f"PASS: q >> p, low acceptance ({prob:.4f})")


def test_residual_all_mass_on_correct_token():
    # p = [0.7, 0.3], q = [0.3, 0.7], residual = [0.4, 0] so token 1 never sampled
    p = torch.tensor([0.7, 0.3])
    q = torch.tensor([0.3, 0.7])
    counts = [0, 0]
    for _ in range(1000):
        counts[sample_residual(p, q)] += 1
    assert counts[1] == 0, f"token 1 should never be sampled, got {counts[1]}"
    print(f"PASS: residual puts all mass on token 0 ({counts[0]}/1000)")


def test_residual_matches_p_when_disjoint():
    # p and q have no overlap so residual == p
    p = torch.zeros(4); p[0] = 0.6; p[1] = 0.4
    q = torch.zeros(4); q[2] = 0.5; q[3] = 0.5
    counts = [0, 0, 0, 0]
    for _ in range(5000):
        counts[sample_residual(p, q)] += 1
    ratio = counts[0] / (counts[0] + counts[1])
    assert abs(ratio - 0.6) < 0.05, f"expected ~0.6, got {ratio:.3f}"
    print(f"PASS: residual matches p when disjoint (token 0 ratio: {ratio:.3f}, expected 0.6)")


if __name__ == "__main__":
    test_p_equals_q_always_accepts()
    test_p_dominates_always_accepts()
    test_q_dominates_often_rejects()
    test_residual_all_mass_on_correct_token()
    test_residual_matches_p_when_disjoint()
    print("\nall tests passed")
