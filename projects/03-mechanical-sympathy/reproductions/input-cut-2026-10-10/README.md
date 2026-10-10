<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Reproduce the input-cut-only real-data rollout

See [REPRODUCE.md](REPRODUCE.md) for the concise command and settings.
This directory contains the unpacked reproduction bundle; clone this branch
and change into this directory before running the commands below.

Recorded forward sum: 58.691403893 s; full rollout loop: 58.850663142 s;
median independent forward: 196.189182 ms. BF16 matmult + folding + tiling
without the input cut previously measured 42.58391398 s.

This bundle freezes the locally modified runner from /amogh/tal-easy-wins.
The graph cut runs torch_xla.sync() in a UNet pre-hook, after input concat.
No per-layer segmentation or four-core spatial parallelism is enabled.

Required environment: Trainium2, LNC2, one available logical core, Python 3.12,
and the packages recorded in versions.json, plus the Samudra dependencies.
The original pod exposed one Trainium2 chip on trn2.48xlarge. It used core 0,
eight CPU threads, CPU affinity 48–55 (NUMA 1), runtime 2.34.10.

Use Samudra source based on commit
6b88f2c0204828ef77d41b3ab2733679615f53e8. The measured Samudra commit was
a5fc211a47245178d34c9e816a86e838784e9ce0; its only src changes from that base
were forward_benchmark.py and inference_fusion.py. This bundle includes the
required inference_fusion.py helper; copy it into src/samudra/utils/ of your
checkout if absent. The runner imports the bundled forward_timing.py instead
of Samudra's added forward_benchmark.py.

Copy these large artifacts separately (excluded from this bundle):

- Checkpoint: /amogh/samudra-one-degree/checkpoints/onedeg/ema_ckpt.pt
  SHA-256 ff6160cb7e2a2b5b8ee2c5731c16bd114a1bb5a267245cc5ed766fb00c7f797a
- Prepared inputs: /amogh/samudra-one-degree/outputs/forward_benchmark/eval_case.pt
  SHA-256 7251e96fe08c1d3c794522020ab10d579c97d1bcd6ce98ab951efd9c0746198b

The prepared case contains the actual initial state, mask/context and all
299 forcing windows. Using it avoids downloading OM4 again for this replay.
eval.replay.yaml preserves the original case specification, including its
original source paths and data_root, so the runner's case guard passes even
when the checkpoint, prepared case, source checkout and working directory
have moved. Those original source locations are not read when this case exists.

```bash
export SAMUDRA_ROOT=/path/to/Samudra
export CHECKPOINT=/path/to/ema_ckpt.pt
export PREPARED_CASE=/path/to/eval_case.pt
export NEURON_PYTHON=/path/to/neuron-venv/bin/python
export NEURON_RT_VISIBLE_CORES=0  # choose a free logical core
export CPU_AFFINITY=48-55        # change for the destination NUMA/cpuset
cp inference_fusion.py "$SAMUDRA_ROOT/src/samudra/utils/inference_fusion.py"
./run.sh
```

The model is in eval mode with gradients and activation checkpointing disabled;
FP32 input/output tensors, BF16 matmult autocast, 18 folded BatchNorm pairs.
Only unet.layers.0 and unet.layers.16 are tiled. Grid 180×360, batch 1,
two input/two output steps, 299 autoregressive calls / 598 forecast records,
2014-10-20 through 2022-12-24 output times. Weights, initial state and forcing
are device-resident before timing; three warmup calls and 20 independent
forward samples precede the full rollout. Setup, compilation, transfers,
history updates, output writes and metrics are outside forward timing.
No forecast metrics or prediction outputs are produced.

Use --segments in place of --input-cut to reproduce the separate segmented
variant (14.081677032 s in its recorded BF16 + tiling run). This adds cuts after
all 17 UNet layers and is a different graph configuration. The standalone
trainium/bench.py --segments 1 test uses seeded random inputs rather than these
real autoregressive inputs; its published FP32 numbers are a separate workload.

FP32 compiler arithmetic is selected with --auto-cast=none, but run.sh sets
BF16 flags explicitly: edit its NEURON_CC_FLAGS line for an FP32 comparison.
Exact latency depends on compiler/runtime, hardware and concurrent load.
