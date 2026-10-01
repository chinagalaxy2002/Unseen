# HS-DQ-CGP anti-collapse revision

## Diagnosis from the stopped run

By epoch 63, the first implementation had converged without numerical errors,
but its routing mechanism had degenerated:

- 11 of 16 bases had zero average usage;
- cross-level JS reached 0.6931, approximately its maximum;
- adjacent-point JS was effectively zero;
- point correction magnitude fell to 0.54% of level-logit magnitude;
- the same five basis usage values remained almost unchanged after epoch 2.

The original entropy difference objective rewarded deterministic routing. The
direct level embedding then supplied a shortcut that solved this objective by
assigning a fixed basis identity to every temporal scale. Once the level
softmax saturated, the bounded point correction no longer changed the final
probabilities.

## Revision

1. The level embedding remains in temporal binding, where scale is required,
   but the default basis router uses normalized prototype, normalized query,
   and their multiplicative interaction. The old embedding input remains an
   explicit ablation option.
2. Level and point router output layers start near zero. Level logits are
   bounded with a scaled tanh before softmax.
3. Point-router logits are mean-centered over valid points in each level. The
   point branch can express occurrence deviations but cannot reproduce the
   level branch.
4. Routing regularization is split into:
   - KL load balancing of equally weighted level routes against uniform basis
     usage;
   - a moderate per-route entropy target, avoiding both uniform and one-hot
     routing;
   - a minimum per-level batch-usage entropy, preventing a fixed basis identity
     for a level.
5. Utilization is measured over `[B,L,N]`, not over dense points, so short and
   long pyramid levels contribute equally.
6. New diagnostics record cross-query level JS, point-effect JS, effective
   basis count, usage entropy, route entropy, and every anti-collapse loss
   component.
7. The variant launcher uses ReduceLROnPlateau because the reused trainer calls
   `scheduler.step(training_loss)`.

## Acceptance checks for the next run

Inspect the first 5--10 epochs before committing to a long run:

- effective basis count should remain well above 5 and should not collapse in
  one or two epochs;
- cross-level JS should not immediately saturate at 0.6931;
- cross-query level JS should become non-zero, showing query-dependent routing;
- point-effect JS should become measurable while adjacent JS stays below
  cross-level JS;
- point correction should remain bounded and ideally contribute roughly
  1--20% of level-logit magnitude;
- only enable smoothness loss if adjacent JS becomes excessive.

The stopped run and its best checkpoint remain under
`outputs/flashvtg_hs_dq_cgp_seed2024`; the revised launcher writes to
`outputs/flashvtg_hs_dq_cgp_anticollapse_seed2024` by default.
