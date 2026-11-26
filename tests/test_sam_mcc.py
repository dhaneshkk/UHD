"""Regression tests for SAM optimizer and MCC metrics. Run: python tests/test_sam_mcc.py"""
import copy, math, os, random, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from uhd.sam import SAM, sam_update
from uhd.metrics import evaluate_mcc, mcc_from_counts

torch.manual_seed(0)

# ---------- 1. SAM step == closed-form definition (plain SGD, no momentum) ----------
for adaptive in (False, True):
    w0 = torch.randn(5, dtype=torch.float64)
    A = torch.randn(5, 5, dtype=torch.float64); A = A @ A.T + torch.eye(5, dtype=torch.float64)
    f = lambda w: 0.5 * w @ A @ w + torch.sin(w).sum()
    grad = lambda w: A @ w + torch.cos(w)
    rho, lr = 0.1, 0.05
    g = grad(w0)
    e = (w0 ** 2 * g if adaptive else g) * rho / ((w0.abs() * g if adaptive else g).norm())
    expected = w0 - lr * grad(w0 + e)
    p = nn.Parameter(w0.clone())
    sam = SAM(torch.optim.SGD([p], lr=lr), rho=rho, adaptive=adaptive)
    f(p).backward()
    sam.first_step()
    if not adaptive:
        assert abs((p.detach() - w0).norm().item() - rho) < 1e-12, "perturbation norm must equal rho"
    f(p).backward()
    sam.second_step()
    assert torch.allclose(p.detach(), expected, atol=1e-12), (adaptive, p, expected)
print("SAM closed-form OK (SAM and ASAM)")

# ---------- 2. sam_update: GradScaler on == off; BN stats frozen on pass 2; restore ----------
def make():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 1))
x, y = torch.randn(16, 4), torch.randn(16, 1)
results = []
for use_scaler in (False, True):
    m = make(); m.train()
    sam = SAM(torch.optim.AdamW(m.parameters(), lr=1e-2), rho=0.05)
    scaler = torch.amp.GradScaler("cpu", enabled=use_scaler, init_scale=1024.0)
    fwd_calls = []
    def fwd():
        fwd_calls.append(1)
        out = m(x); return ((out - y) ** 2).mean(), {}
    loss, _ = fwd()
    bn = m[1]; rm_after_pass1 = bn.running_mean.clone()
    ok = sam_update(sam, scaler, loss, fwd, m, list(m.parameters()), grad_clip_norm=5.0)
    assert ok and len(fwd_calls) == 2 and not sam._e_w
    assert torch.equal(bn.running_mean, rm_after_pass1), "BN running stats changed on perturbed pass"
    assert bn.momentum == 0.1, "BN momentum not restored"
    results.append([p.detach().clone() for p in m.parameters()])
for a, b in zip(*results):
    assert torch.allclose(a, b, atol=1e-6), "scaler path differs from fp32 path"
print("sam_update OK (scaler == no scaler, BN frozen + restored)")

# ---------- 3. non-finite first-pass grads: skip, weights untouched, scale backed off ----------
m = make(); sam = SAM(torch.optim.SGD(m.parameters(), lr=0.1), rho=0.05)
scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
before = [p.detach().clone() for p in m.parameters()]
loss = (m(x) * float("inf")).mean()
ok = sam_update(sam, scaler, loss, lambda: (loss, {}), m, list(m.parameters()))
assert not ok and scaler.get_scale() == 512.0 and not sam._e_w
assert all(torch.equal(a, b.detach()) for a, b in zip(before, m.parameters()))
# exception during pass 2 never leaves weights perturbed
m = make(); sam = SAM(torch.optim.SGD(m.parameters(), lr=0.1), rho=0.05)
before = [p.detach().clone() for p in m.parameters()]
loss = ((m(x) - y) ** 2).mean()
def boom(): raise RuntimeError("boom")
try:
    sam_update(sam, None, loss, boom, m, list(m.parameters()))
except RuntimeError:
    pass
assert all(torch.equal(a, b.detach()) for a, b in zip(before, m.parameters())) and m[1].momentum == 0.1
print("sam_update skip / exception safety OK")

# ---------- 4. scheduler interplay + checkpoint interchange ----------
m = make(); sam = SAM(torch.optim.AdamW(m.parameters(), lr=1e-2), rho=0.05)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(sam, T_max=10)
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    for _ in range(3):
        sam.zero_grad(); loss = ((m(x) - y) ** 2).mean()
        sam_update(sam, None, loss, lambda: (((m(x) - y) ** 2).mean(), {}), m, list(m.parameters()))
        sched.step()
assert not [w for w in wlist if "lr_scheduler" in str(w.message)], [str(w.message) for w in wlist]
assert abs(sam.base_optimizer.param_groups[0]["lr"] - sched.get_last_lr()[0]) < 1e-15
sd = sam.state_dict()
plain = torch.optim.AdamW(make().parameters(), lr=1e-2); plain.load_state_dict(sd)   # SAM ckpt -> plain
sam2 = SAM(torch.optim.AdamW(make().parameters(), lr=1e-2)); sam2.load_state_dict(plain.state_dict())  # plain -> SAM
assert sam2.param_groups is sam2.base_optimizer.param_groups
print("scheduler (no warning) + state_dict interchange OK")

# ---------- 5. MCC ----------
rng = random.Random(0)
for _ in range(2000):
    tp, fp, fn, tn = (rng.randint(0, 50) for _ in range(4))
    v = mcc_from_counts(tp, fp, fn, tn)
    assert -1.0 - 1e-12 <= v <= 1.0 + 1e-12
    yt = [1] * (tp + fn) + [0] * (fp + tn); yp = [1] * tp + [0] * fn + [1] * fp + [0] * tn
    try:
        from sklearn.metrics import matthews_corrcoef
        assert abs(v - matthews_corrcoef(yt, yp)) < 1e-9
    except ImportError:
        pass
assert mcc_from_counts(10, 0, 0, 90) == 1.0 and mcc_from_counts(0, 10, 90, 0) == -1.0

gt = [{"boxes": torch.tensor([[0.2, 0.2, 0.2, 0.3], [0.7, 0.6, 0.2, 0.4]]), "labels": torch.tensor([0, 0])}]
perfect = [[(0.9, 0, gt[0]["boxes"][0]), (0.8, 0, gt[0]["boxes"][1])]]
r = evaluate_mcc(perfect, gt, 1, score_thresh=0.15, grid_hw=(8, 8))
assert r["mcc"] == 1.0 and r["precision"] == r["recall"] == r["f1"] == 1.0
r = evaluate_mcc([[(0.1, 0, gt[0]["boxes"][0])]], gt, 1, score_thresh=0.15)  # below threshold
assert r["mcc"] == 0.0 and r["recall"] == 0.0 and r["mcc_cells_fn"] == 2
r = evaluate_mcc([[(0.9, 0, gt[0]["boxes"][0]), (0.9, 0, torch.tensor([0.9, 0.9, 0.1, 0.1]))]], gt, 1, 0.15)
assert r["precision"] == 0.5 and r["recall"] == 0.5 and 0 < r["mcc"] < 1, r
# right cell, wrong box -> cell TP but box FP + FN
r = evaluate_mcc([[(0.9, 0, torch.tensor([0.2, 0.2, 0.02, 0.02]))]], gt[:1], 1, 0.15)
assert r["mcc_cells_tp"] == 1 and r["precision"] == 0.0
print("MCC OK")
print("ALL PASS")
