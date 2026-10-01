# FlashVTG HS-DQ-CGP

This is an isolated implementation of hierarchical scale-conditioned
compositional prompt generation for FlashVTG. The source baseline and the
previous point-wise DQ-CGP directory are not changed.

The hierarchy is:

```text
point-wise temporal binding
  -> query-conditioned per-level prototype
  -> bounded [B, L, N] prototype-query interaction routing
  -> centered 0.1 * tanh(point residual routing)
  -> shared prompt bank + token attention
  -> point-wise FRF[prompt, text, context, local feature]
  -> fixed 0.05 residual
```

Default hyperparameters in `model_config.py` are: 16 bases, 6 prompt tokens,
level-router width 256, point-router width 128, temperature 1.0, correction
bound 0.10, bounded level logits at 2.0, beta 0.05, locality strength 0.5,
binding loss 0.05, route loss 0.005, and smoothness loss 0.0.

The anti-collapse revision intentionally excludes the direct level embedding
from basis routing (it remains in temporal binding), starts both routers near
uniform, and uses three separate routing constraints: global basis load
balance, a moderate per-route entropy target, and a per-level batch-utilization
floor. Point-router logits are centered within each level so the point branch
cannot silently become another level router.

Run the dedicated launcher from the repository root:

```bash
bash models/flashvtg_hs_dq_cgp_gmr/train_hs_dq_cgp.sh
```

To warm-start from a baseline FlashVTG checkpoint, add `--resume PATH` to that
command. The loader accepts a baseline checkpoint only when every missing
parameter belongs to `dq_cgp.*`.

The loss log records scale JS, adjacent JS, cross-query level JS, point effect
JS, correction magnitude, effective basis count, route and usage entropy, each
anti-collapse loss component, and one `loss_dq_cgp_basis_usage_XX` scalar per
basis. In the intended first run, `loss_dq_cgp_smooth` is recorded but has zero
weight; set its config value to 0.005--0.01 only if adjacent routing remains too
variable. The launcher uses a loss-aware ReduceLROnPlateau scheduler because
the reused trainer passes training loss into `scheduler.step(...)`.
