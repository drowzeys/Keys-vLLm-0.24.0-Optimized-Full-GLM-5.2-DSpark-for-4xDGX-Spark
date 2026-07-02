# vLLm 0.23.1rc1 - GLM-5.2 Quantrio INT4/INT8 mixed + DSpark speculator (RedHat AI) on a 4× DGX Spark / GB10 cluster

**First known serve of GLM-5.2 with the upstream vLLM DSpark speculative-decoding
stack (PRs [#46995](https://github.com/vllm-project/vllm/pull/46995) +
[#47093](https://github.com/vllm-project/vllm/pull/47093)) and RedHat AI's
[`GLM-5.2-speculator.dspark`](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark)
draft on consumer Blackwell (sm_121a) — ~24 hours after the PRs merged upstream.**

Target: `QuantTrio/GLM-5.2-Int4-Int8Mix` (full, non-pruned 4-bit-expert / 8-bit-attention
quant — the only GLM-5.2 that fits 4×128GB unified memory), TP=4 over 200G RoCE.

## Results (2026-07-02, epoch-1 draft, k=7, greedy-class sampling)

| Metric | Value |
| --- | --- |
| Serve status | ✅ coherent (code + reasoning correct) |
| Single-stream decode | **11.3–12.4 tok/s** (true tokens) |
| C2 aggregate | **15.9 tok/s** |
| Mean accepted length (notarized, 857 drafts) | **2.16** |
| Per-position acceptance (1→6) | 0.60 / 0.30 / 0.14 / 0.06 / 0.03 / 0.02 |
| Reference on FP8 target (RedHat, 4×B300) | 3.376 accepted length, pos-1 0.78 |

### The headline finding: quantized-target acceptance transfer

The draft was trained on **FP8** GLM-5.2 hidden states. Serving it against the
**Int4-Int8** QuantTrio target costs ~half its acceptance (pos-1: 0.78 → 0.60;
accepted length 3.38 → 2.16). At k=7 that means ~40 tok/s drafted for ~6 accepted —
enough waste that DSpark currently lands **below** this cluster's MTP k=3 baseline
(15.3 tok/s single-stream, same target, same eager config).

**Implications** (in order of expected payoff):
1. Retrain the draft against QuantTrio hidden states — the
   [speculators](https://github.com/vllm-project/speculators) online-training recipe
   streams hiddens from a live vLLM server (no offline cache needed).
2. `num_speculative_tokens` 7 → 3–4 to cut draft waste at low acceptance.
3. Epoch-2/3 checkpoints (RedHat is publishing per-epoch revisions) — drop-in swaps.

## The build

`vllm-glm52-cuda130:dspark` = the [January full-GLM-5.2 image]
(https://github.com/drowzeys/Keys---Full-GLM-5.2-Quantrio-INT4-INT8-mixed-8bit-Attention-on-4-x-DGX-Spark-GB10-Cluster)
(vLLM 0.23.1rc @ `ab666069` + CUDA-13.0 rebuild + DSA sm12x mods) **+** both DSpark PRs
cherry-picked onto that exact base commit and 3-way merged against the live image tree
(**zero conflicts** — the GLM mods and the DSpark PRs touch disjoint files) **+** one
GB10 shim of ours:

- **Draft KV-dtype decouple** (`patches/dspark_utils_kv_shim.py` region): the upstream
  dspark loader hands the draft the *target's* cache config, so an MLA target
  (`fp8_ds_mla`) makes the draft's FLASH_ATTN/FLASHINFER validation fail with
  `kv_cache_dtype not supported`. Fix: draft gets `cache_dtype="auto"` (bf16).
  Affects ANY MLA-family target + dspark + non-default KV dtype — upstreamable.

Serve command (per-node, mp multi-node — see `recipe/`):

```bash
vllm serve <QuantTrio-snapshot> --served-model-name glm-5.2-dspark \
  -tp 4 --nnodes 4 --node-rank N --master-addr 10.100.10.4 --distributed-executor-backend mp \
  --kv-cache-dtype fp8 --max-model-len 32768 --max-num-seqs 2 \
  --gpu-memory-utilization 0.86 --num-gpu-blocks-override 768 --enforce-eager \
  --speculative-config '{"method":"dspark","model":"<RedHatAI-draft-snapshot>","num_speculative_tokens":7,"attention_backend":"FLASH_ATTN","draft_sample_method":"probabilistic"}'
```

## The 13 walls (findings ledger)

Two transplant targets were attempted. The **0.24-port path** (vllm-dspark024, our
DSV4F-DSpark base) fell to a hardware-shaped dead end; the **January-GLM path** served.
Every wall + fix, in order:

| # | Wall | Fix / finding |
|---|---|---|
| 1 | JSON quoting eaten by env→ssh→bash relay | single-quote payloads at the last shell |
| 2 | Draft inherits target's `fp8_ds_mla` KV dtype → FLASH_ATTN invalid | **KV-decouple shim** (upstreamable bug) |
| 3 | deep_gemm asserts `Unsupported architecture` (DSA indexer, nightly) | sm12x Triton bypass port (January technique) |
| 4 | aidendle sm_121a deep_gemm transplant wedges nightly | version-locked: transplants don't cross vLLM generations |
| 5 | `VLLM_DEEP_GEMM_WARMUP=skip` ignored / autotune hang | flashinfer autotuner busy-spins **96% util @ 16W** on sm_121a |
| 6 | flashinfer autotuner spin (0.6.12 AND 0.6.13) | excised at source (`tune_mode=False`) — GB10 images should ship this |
| 7 | latent port bug: `flashmla.py` used alias before import (FI_SPARSE=0 path) | import-order fix — never fired because DSV4F always ran =1 |
| 8 | deep_gemm JIT wants `sm121_*.cuh` includes that don't exist | **sm121→sm120 shim headers** |
| 9 | JIT rejects `#include "quoted"` | angle-bracket house style |
| 10 | JIT instantiates `sm121_*` template symbol | **`#define`-rename at definition time** (shim technique, reusable) |
| 11 | `trtllm_batch_decode_with_kv_cache_mla(kv_scale_format=)` | flashinfer 0.6.12/0.6.13 API skew — signature-adaptive kwarg |
| 12 | XQA MLA on SM120/121: `(bf16, uint8-fp8)` rejected | **hardware-shaped dead end** for bf16-activation models on the 0.24 fp8-XQA path → pivot |
| 13 | NCCL bootstrap death on the January image | launcher must use `/opt/nvidia/nvidia_entrypoint.sh` (its CUDA/IB env setup); bare `bash -lc` + foreign LD_PRELOAD kills PyNccl |

Wall 8–10's sm121 shim set and wall 6's autotuner excision are in `patches/`.

## Contents
- `recipe/` — launchers (the proven exp-launch pair with `DSPARK_DRAFT` support) + serve args.
- `patches/` — draft-KV shim, sm121 include shims, autotuner excision, kwarg-skew fix, transplant Dockerfiles.
- `docs/CAMPAIGN.md` — the full two-path campaign narrative with all launch logs distilled.

## Reproduce
1. Build/obtain the January GLM image (`vllm-glm52-cuda130:full`, see the QuantTrio repo).
2. `git -C vllm checkout ab666069 && git cherry-pick -m1 f5a8d7337 2b753ad20` → 3-way merge the 20 source files onto the image tree (zero conflicts expected) + apply `patches/dspark_utils_kv_shim`.
3. `docker build` with the import gate in `recipe/Dockerfile.023dspark`.
4. Stage `RedHatAI/GLM-5.2-speculator.dspark` (7.1G) on every node.
5. Launch via `recipe/launch-cluster.sh` with `DSPARK_DRAFT=models--RedHatAI--GLM-5.2-speculator.dspark`.

## Attribution
Upstream DSpark: vLLM project + Michael Goin / RedHat AI (speculator checkpoints, speculators library).
Base image lineage: eugr/spark-vllm-docker, lukealonso/b12x, CosmicRaisins sm12x kernels.
DeepSeek DSpark architecture: DeepSeek-AI. Apache-2.0 for our patches; serves MIT GLM-5.2 weights.
