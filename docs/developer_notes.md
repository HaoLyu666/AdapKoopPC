# Developer Notes

## Checkpoint Compatibility

The neural model code has been cleaned up, but some module attribute names intentionally remain close to the original research code. The pretrained checkpoint files store PyTorch `state_dict` keys such as:

```text
mapping
linear1.weight
qt.weight
project0.weight
A.weight
B.weight
C.weight
dec.project0.weight
```

Renaming those module attributes would break direct loading of `checkpoints/true_new_64/epoch8_*.tar`. For this reason:

- Internal temporary variables in `models.py` were renamed for readability.
- Public aliases were added, for example `KoopmanSpace` and `KoopSpace`.
- Legacy-compatible aliases remain, for example `Koop_space` and `decoder`.
- The old misspelled config key `liner_dec` is still emitted by the config bridge, while the runtime attribute is exposed as `linear_decoder`.

This keeps the code cleaner without requiring checkpoint conversion.

## Refactor Boundary

The current open-source-ready path covers the control simulation:

```text
Model checkpoint -> Koopman matrix construction -> MPC optimization -> simulation outputs
```

The original training and prediction-evaluation scripts depended on large HighD `.mat` files and were not moved into the default runnable package.
