"""Regression tests for eval / split / checkpoint-safety fixes. Run: python tests/test_eval_split_safety.py"""
import os, sys, random, tempfile, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from uhd.data import split_items, split_group_key
from uhd.metrics import evaluate_map, nms_per_class, decode_anchor
import uhd.metrics as M
from uhd.utils import safe_torch_load, TRUST_CKPT_ENV

# ---------- 1. split ----------
items = []
for i in range(300):
    cid = f"{i:012d}"
    items.append((f"/d/{cid}.jpg", f"/d/{cid}.txt"))
    if i % 3 == 0:
        items.append((f"/d/dist_{cid}.jpg", f"/d/dist_{cid}.txt"))
for seq in ["frame6", "frameA", "frameB", "frame7"]:
    for f in range(40):
        items.append((f"/d/{seq}_{f:06d}.jpg", f"/d/{seq}_{f:06d}.txt"))
assert split_group_key("/d/dist_000000075375.jpg") == split_group_key("/d/000000075375.png") == "coco:000000075375"
assert split_group_key("/d/frameA_000088.jpg") == "seq:frameA"
assert split_group_key("/d/frame6_000286.jpg") == "seq:frame6"
tr, va = split_items(items, 0.8, 0.2, seed=42)
gk = lambda its: {split_group_key(p) for p, _ in its}
assert not (gk(tr) & gk(va)), "group leakage"
assert set(tr).isdisjoint(va)
assert abs(len(va) - int(len(items) * 0.2)) <= 40, len(va)
# legacy mode demonstrably leaks
tr_i, va_i = split_items(items, 0.8, 0.2, seed=42, mode="image")
print(f"image-mode leaking groups: {len(gk(tr_i) & gk(va_i))}  group-mode: 0")
# determinism regardless of input order
shuf = items[:]; random.Random(1).shuffle(shuf)
assert split_items(shuf, 0.8, 0.2, 42, verbose=False) == (tr, va)
assert split_items(shuf, 0.8, 0.2, 42, mode="image") == (tr_i, va_i)
# regex override
assert split_group_key("/d/cam03_2026_000123.jpg", r"^(cam\d+)") == "re:cam03"
# giant-group guard
bad = [(f"/d/img_{i:05d}.jpg", "") for i in range(100)]
try:
    split_items(bad, 0.8, 0.2, 42); raise AssertionError("expected error")
except ValueError as e:
    assert "split-group-regex" in str(e)
print("split OK")

# ---------- 2. mAP truncation ----------
torch.manual_seed(0)
targets, preds = [], []
for i in range(200):
    gt = torch.tensor([[0.5, 0.5, 0.3, 0.5]])
    targets.append({"boxes": gt, "labels": torch.tensor([0])})
    # half the TPs are low-confidence, plus a few low FPs
    s = 0.9 if i % 2 == 0 else 0.05
    preds.append([(s, 0, gt[0].clone()), (0.02, 0, torch.tensor([0.1, 0.1, 0.1, 0.1]))])
full = evaluate_map(preds, targets, 1)["mAP@0.5"]
trunc = evaluate_map([[d for d in p if d[0] >= 0.15] for p in preds], targets, 1)["mAP@0.5"]
print(f"AP at eval thresh 0.001: {full:.3f}   truncated at 0.15: {trunc:.3f}")
assert full > trunc + 0.3
# no-GT class excluded from mean
t2 = [{"boxes": torch.tensor([[0.5,0.5,0.2,0.2]]), "labels": torch.tensor([0])}]
p2 = [[(0.9, 0, torch.tensor([0.5,0.5,0.2,0.2]))]]
r = evaluate_map(p2, t2, num_classes=2)
assert abs(r["mAP@0.5"] - 1.0) < 1e-6, r
print("mAP OK", r)

# ---------- 3. NMS equivalence ----------
for trial in range(20):
    n = 300
    bx = torch.rand(n, 4) * torch.tensor([1, 1, 0.3, 0.3]) + torch.tensor([0, 0, 0.02, 0.02])
    sc = torch.rand(n)
    cl = torch.randint(0, 3, (n,))
    boxes = [(float(sc[i]), int(cl[i]), bx[i]) for i in range(n)]
    fast = nms_per_class(boxes, 0.5)
    saved = M._tv_batched_nms; M._tv_batched_nms = None
    slow = nms_per_class(boxes, 0.5)
    M._tv_batched_nms = saved
    kf = sorted((round(s, 6), c) for s, c, _ in fast); ks = sorted((round(s, 6), c) for s, c, _ in slow)
    assert kf == ks, (trial, len(kf), len(ks))
    assert [d[0] for d in fast] == sorted([d[0] for d in fast], reverse=True)
print("NMS OK (torchvision == python reference)")

# ---------- 4. decode_anchor pre_nms_topk ----------
na, nc, H, W = 8, 1, 16, 16
anchors = torch.rand(na, 2) * 0.5 + 0.05
pred = torch.randn(2, na * (5 + nc), H, W)
a = decode_anchor(pred, anchors, nc, conf_thresh=0.001, nms_thresh=1.01)  # nms off
b = decode_anchor(pred, anchors, nc, conf_thresh=0.001, nms_thresh=1.01, pre_nms_topk=100)
assert len(b[0]) == 100 and len(a[0]) > 100
assert sorted(d[0] for d in a[0])[-100:] == sorted(d[0] for d in b[0])
print("pre_nms_topk OK", len(a[0]), "->", len(b[0]))

# ---------- 5. safe_torch_load ----------
class Evil:
    def __reduce__(self):
        return (os.system, ("echo PWNED > /tmp/pwned",))
d = tempfile.mkdtemp()
ok = os.path.join(d, "ok.pt"); bad = os.path.join(d, "bad.pt")
torch.save({"model": {"w": torch.ones(2)}, "arch": "cnn", "best_map": 0.4, "class_ids": [0], "ema": None,
            "optimizer": torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1).state_dict()}, ok)
torch.save({"x": Evil()}, bad)
assert safe_torch_load(ok)["arch"] == "cnn"
if os.path.exists("/tmp/pwned"): os.remove("/tmp/pwned")
try:
    safe_torch_load(bad); raise AssertionError("expected refusal")
except RuntimeError as e:
    assert TRUST_CKPT_ENV in str(e)
assert not os.path.exists("/tmp/pwned"), "payload executed!"
print("safe_torch_load OK (payload blocked)")
print("ALL PASS")
