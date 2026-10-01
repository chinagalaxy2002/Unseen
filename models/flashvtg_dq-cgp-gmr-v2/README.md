# FlashVTG-DQ-CGP implementation

This directory contains the candidate-wise DQ-CGP extension used by the
repository-level training and evaluation scripts.

FlashVTG has no DETR decoder-query axis. The implementation therefore treats
every dense FlashVTG pyramid point as a native candidate and applies:

```text
FlashVTG fusion
  -> native dense candidates
  -> candidate-local temporal binding
  -> RCG basis routing
  -> BPS/FRF residual refinement
  -> original FlashVTG classification and coordinate heads
```

The main files are:

- `dq_cgp.py`: temporal binding, routing, basis prompts, and residual fusion;
- `model.py`: integration into FlashVTG and auxiliary DQ-CGP losses;
- `model_config.py`: the released hyperparameters;
- `run.py`: training/inference launcher reusing the released FlashVTG loops;
- `tests/test_dq_cgp.py`: focused unit tests.

The matched FlashVTG baseline is kept under `models/flash_vtg_gmr` and is not
modified. See the repository root `README.md` for exact reproduction commands.
