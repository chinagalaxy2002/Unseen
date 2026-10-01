# FlashVTG DQ-CGP-v3

This is an isolated DQ-CGP-v3 experiment directory. It does not alter the
released FlashVTG tree, the previous point-wise DQ-CGP implementation, or the
two HS-DQ-CGP experiment directories.

V3 addresses the two observed HS-DQ-CGP failure modes: diffuse near-uniform
level routes and a point residual route that vanished in logit space.

```text
point-wise temporal binding
  -> query-conditioned global level prototype c_bar[l]
  -> query-conditioned local occurrence prototype c_local[l,p]
  -> sparse differentiable top-4 level routing w_level[l]
  -> sparse differentiable top-4 point routing u[l,p]
  -> w[l,p] = 0.9 * w_level[l] + 0.1 * u[l,p]
  -> shared prompt bank + prompt-token attention
  -> point-wise FRF[prompt, text, context, local feature]
  -> fixed 0.05 residual -> original FlashVTG heads
```

`torch.topk` fixes a sparse support in the forward pass; softmax gradients are
retained for the logits inside that support. The point mixture is performed in
probability space, so `0.1` means exactly a 10% point-route contribution, not
a small and potentially ineffective logit offset.

The training criterion keeps binding supervision and adds two route terms:

- global basis load balancing plus a conditional entropy target over the
  top-4 support;
- a weak relation loss which matches same-level route divergence to detached
  temporal-attention divergence. It encourages occurrence specialization only
  when the point-wise temporal evidence differs; it is not unconditional
  adjacent smoothing.

Default v3 settings are 16 bases, 6 prompt tokens, top-k 4, probability mix
ratio 0.10, local-prototype radius 2 stride units, fixed residual beta 0.05,
and relation-loss coefficient 0.02.

Run from the repository root:

```bash
bash scripts/train_flashvtg_dq_cgp_v3.sh
```

The launcher writes to `outputs/flashvtg_dq_cgp_v3_topk4_seed2024` by default.
Override `DQ_V3_GPU`, `DQ_V3_EPOCHS`, or `DQ_V3_OUTPUT` for an experiment.

For the exact training and Standard-test reproduction commands, logs, and
metric tables, see the repository root [README](../../README.md).

Key diagnostics are `level_active_basis_count`, `point_active_basis_count`,
point correction total-variation ratio, point-effect JS, relation route JS,
relation attention JS, basis utilization, and level/query routing JS.
