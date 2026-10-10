<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Reproduce the input-cut-only rollout

For the **input-cut-only run that slowed down BF16 + tiling**, use these settings:

| Setting | Value |
|---|---|
| Trainium allocation | **One logical core**, LNC2 |
| Compiler | `--target=trn2 --auto-cast=matmult --auto-cast-type=bf16 --logical-nc-config=2` |
| CPU | Eight threads; original affinity `48–55` |
| Tiling | 30 rows, minimum `64800` pixels |
| BatchNorm folding | Enabled, 18 pairs |
| Graph cuts | **After input concatenation only**; no per-layer segmentation |
| Workload | Real checkpoint and prepared OM4 inputs, 299 autoregressive calls |
| Sampling | Three warmups, 20 independent samples, then full rollout |

The recorded forward time was **58.69 s**, versus **42.58 s** without the input cut.

I packaged the exact modified runner and a portable replay config:

[Reproduction files and detailed setup](README.md)

On the destination system, clone this branch and change into
`projects/03-mechanical-sympathy/reproductions/input-cut-2026-10-10`, then run:

```bash
export SAMUDRA_ROOT=/path/to/Samudra
export CHECKPOINT=/path/to/ema_ckpt.pt
export PREPARED_CASE=/path/to/eval_case.pt
export NEURON_PYTHON=/path/to/neuron-venv/bin/python

export NEURON_RT_VISIBLE_CORES=0  # choose a free core
export CPU_AFFINITY=48-55        # adjust to destination CPU affinity

cp inference_fusion.py \
  "$SAMUDRA_ROOT/src/samudra/utils/inference_fusion.py"

./run.sh
```

The script invokes:

```bash
--backend neuron --threads 8 --warmup 3 --single-repeats 20 \
--tile-rows 30 --tile-min-pixels 64800 \
--fold-batch-norm --input-cut
```

Copy the checkpoint and prepared inputs separately from:

```text
/amogh/samudra-one-degree/checkpoints/onedeg/ema_ckpt.pt
/amogh/samudra-one-degree/outputs/forward_benchmark/eval_case.pt
```

Match **Python 3.12, PyTorch 2.9.1, torch-xla 2.9.0, torch-neuronx 2.9.0.2.15.32035+de43f57c, and neuronx-cc 2.27.5334.0+f702b353**. The bundle README includes source revisions and artifact hashes.

The **14.08 s segmented run** is a separate configuration using `--segments` instead of `--input-cut`. The published FP32 input-cut measurements also used a separate, random-input benchmark.
