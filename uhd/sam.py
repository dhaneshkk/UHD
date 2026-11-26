"""Sharpness-Aware Minimization (Foret et al., 2021) and Adaptive SAM (Kwon et al., 2021).

SAM minimizes max_{||e|| <= rho} L(w + e). Each update takes two forward/backward passes:
  1. g = dL/dw at w;  e = rho * g / ||g||   (ASAM: e = rho * w^2 * g / ||w * g||)
  2. g' = dL/dw at w + e;  restore w;  base_optimizer.step() using g'.

`SAM` wraps any torch optimizer and shares its param_groups, so LR schedulers, checkpoint
state_dict()/load_state_dict() and GradScaler (via `.base_optimizer`) keep working.
"""
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class SAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05, adaptive: bool = False, eps: float = 1e-12):
        if rho <= 0:
            raise ValueError(f"SAM rho must be > 0, got {rho}")
        self.base_optimizer = base_optimizer
        self.rho = float(rho)
        self.adaptive = bool(adaptive)
        self.eps = float(eps)
        # Share param_groups (and their 'lr') with the base optimizer so schedulers act on it.
        super().__init__(base_optimizer.param_groups, dict(rho=self.rho, adaptive=self.adaptive))
        self.param_groups = self.base_optimizer.param_groups
        self._e_w: Dict[torch.nn.Parameter, torch.Tensor] = {}  # param -> saved original weights

    def _params_with_grad(self) -> List[torch.nn.Parameter]:
        return [p for g in self.param_groups for p in g["params"] if p.grad is not None]

    @torch.no_grad()
    def grad_norm(self) -> torch.Tensor:
        params = self._params_with_grad()
        if not params:
            return torch.tensor(0.0)
        dev = params[0].device
        norms = [((p.abs() if self.adaptive else 1.0) * p.grad).norm(p=2).to(dev) for p in params]
        return torch.stack(norms).norm(p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> torch.Tensor:
        """Move weights to w + e (the approximate worst case within the rho-ball)."""
        if self._e_w:
            raise RuntimeError("SAM.first_step called twice without restore().")
        gnorm = self.grad_norm()
        scale = self.rho / (gnorm + self.eps)
        for p in self._params_with_grad():
            e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
            self._e_w[p] = p.detach().clone()  # original w, restored exactly (not w + e - e)
            p.add_(e_w)
        if zero_grad:
            self.zero_grad(set_to_none=True)
        return gnorm

    @torch.no_grad()
    def restore(self) -> None:
        """Return weights to w (gradients from the perturbed point are kept)."""
        for p, w in self._e_w.items():
            p.copy_(w)
        self._e_w.clear()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        self.restore()
        self.base_optimizer.step()
        self._opt_called = True  # tell LR schedulers that a step happened (they watch the wrapper)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def step(self, closure: Optional[Callable] = None):
        """Full SAM step with a closure that recomputes loss and calls backward()."""
        if closure is None:
            raise RuntimeError("SAM.step requires a closure; use first_step()/second_step() or sam_update().")
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    # Checkpoints store the base optimizer state, so a SAM run can resume from / into plain runs.
    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups


def _set_bn_momentum(model: torch.nn.Module, saved: Optional[Dict[torch.nn.Module, Optional[float]]] = None):
    """Freeze BN running stats for the perturbed pass (saved=None) or restore them."""
    if saved is None:
        saved = {}
        for m in model.modules():
            if isinstance(m, _BatchNorm):
                saved[m] = m.momentum
                m.momentum = 0.0
        return saved
    for m, mom in saved.items():
        m.momentum = mom
    return None


def _all_finite(params: Iterable[torch.nn.Parameter]) -> bool:
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def sam_update(
    sam: SAM,
    scaler,
    loss: torch.Tensor,
    forward_loss: Callable[[], Tuple[torch.Tensor, Dict]],
    model: torch.nn.Module,
    clip_params: List[torch.nn.Parameter],
    grad_clip_norm: float = 0.0,
) -> bool:
    """One SAM update given the loss already computed at w. Returns False if the step was skipped.

    Works with or without an enabled GradScaler. Gradients of the first pass are unscaled by hand
    (GradScaler.unscale_ may only run once per step), the second pass uses the scaler normally,
    and weights are restored before scaler.step(), so a skipped (inf) step never leaves the model
    at the perturbed point. Grad clipping applies to the second-pass gradient that is stepped.
    """
    use_scaler = scaler is not None and scaler.is_enabled()
    # ---- pass 1: gradient at w ----
    if use_scaler:
        scaler.scale(loss).backward()
        inv = 1.0 / float(scaler.get_scale())
        for p in sam._params_with_grad():
            p.grad.mul_(inv)
        if not _all_finite(sam._params_with_grad()):
            sam.zero_grad(set_to_none=True)
            scaler.update(float(scaler.get_scale()) * float(scaler.get_backoff_factor()))
            return False
    else:
        loss.backward()
        if not _all_finite(sam._params_with_grad()):
            sam.zero_grad(set_to_none=True)
            return False
    sam.first_step(zero_grad=True)

    # ---- pass 2: gradient at w + e (BN running stats frozen) ----
    bn_saved = _set_bn_momentum(model)
    try:
        loss2, _ = forward_loss()
        if not torch.isfinite(loss2):
            sam.restore()
            sam.zero_grad(set_to_none=True)
            return False
        if use_scaler:
            scaler.scale(loss2).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(sam.base_optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm)
            sam.restore()
            scaler.step(sam.base_optimizer)
            scaler.update()
            sam._opt_called = True
        else:
            loss2.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(clip_params, grad_clip_norm)
            sam.second_step()
    finally:
        if sam._e_w:  # exception mid-pass: never leave weights perturbed
            sam.restore()
        _set_bn_momentum(model, bn_saved)
    return True
