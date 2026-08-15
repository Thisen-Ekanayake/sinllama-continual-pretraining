# MI300X training performance: what we changed and why

Engineering record for the UltraChat-Sinhala SFT run (`sft/`), 2026-08-15.
Net result on one MI300X VF (192 GB):

| | before | after |
|---|---|---|
| step time (B=8 x accum 8, seq 2048) | 13.5 s | **7.90 s** |
| effective throughput | 195 TFLOPS | **340 TFLOPS** |
| peak VRAM | 176 GiB (96% of usable) | **131 GiB (72%)** |
| projected epoch | 12.6 h | **7.35 h** |
| projected 3-epoch cap | 38 h | **22 h** |

Training semantics are unchanged: `total_flos` is identical across stacks and
`train_loss` matches to four decimals (2.30094 -> 2.30075).

---

## 1. The baseline

The first real run on `venv/` (torch 2.5.1+rocm6.2) logged **13.5 s/step** and
`total_flos = 100,766,645 GF` over 517 s = 195 TFLOPS, roughly 15% of the
chip's bf16 peak. Slow enough that a 3-epoch run would take 38 hours, so it was
worth understanding before committing the GPU time.

## 2. Method: measure, don't guess

`sft/diag_speed.py` micro-benchmarks the four things that could plausibly be
eating the step, using the exact shapes this model runs:

1. bf16 GEMM ceiling at 4096 / 14336 / 139336 output widths
2. SDPA fwd+bwd under each backend (`flash`, `mem_efficient`, `math`) plus the
   default dispatch, so a silent fallback is visible
3. `lm_head` + cross-entropy over the 139,336-token vocabulary, standard vs Liger
4. a real 2-layer `LlamaForCausalLM` fwd+bwd, extrapolated to 32 layers, to
   confirm the model itself accounts for the measured step time

Run it any time the environment changes:

```bash
python sft/diag_speed.py
```

## 3. Finding 1 — attention was running the O(L^2) math kernel

On torch 2.5.1+rocm6.2, at B=8 / S=2048:

```
default dispatch  48.29 ms      flash 48.37     mem_efficient 48.76     math 49.82
```

All four within 3% — the backend switch did nothing because only one kernel
existed. That work is ~1,650 GFLOP, so 48.3 ms = **34 TFLOPS** against a
535 TFLOPS GEMM ceiling. `attn_implementation: sdpa` had been quietly running
math the whole time, which also explains the memory: the math path materializes
the full `8x32x2048x2048` score matrix (2.1 GB) per layer and keeps it for
backward.

At the real ~1,200-token batch length this was **~4.3 s of every 13.5 s step**.

## 4. Finding 2 — the vocabulary size was pathological for hipBLASLt

```
[16384 x 4096] @ [4096 x   4096]     1.03 ms   535 TFLOPS
[16384 x 4096] @ [4096 x  14336]     3.40 ms   566 TFLOPS
[16384 x 4096] @ [4096 x 139336]   353.32 ms    53 TFLOPS   <- 10x slower
```

139,336 = 2^3 x 17,417, and 17,417 is prime, so `139336 mod 128 = 72`. The N
dimension cannot tile on any sensible boundary and the old hipBLASLt fell back
to a bad kernel configuration for every LM-head GEMM.

We considered padding the vocab to 139,520 (= 128 x 1090), the standard Megatron
trick. **Do not** — see section 6.

## 5. The fix: newer torch/ROCm, installed side by side

The host runs ROCm 7.x while the venv was on 6.2, so the newer AOTriton SDPA
kernels and hipBLASLt were simply unused. Installed in a **second venv** so the
working one stayed intact and rollback is one command:

```bash
source venv/bin/activate && pip freeze > venv-rocm62-freeze.txt && deactivate

python3 -m venv venv-rocm7
source venv-rocm7/bin/activate
pip install -U pip wheel
pip install --index-url https://repo.amd.com/rocm/whl/gfx94X-dcgpu/ torch   # gfx94X = MI300 series
pip install transformers==4.51.3 peft datasets accelerate safetensors sentencepiece pyyaml wandb
pip install "liger-kernel==0.5.10"
```

Landed on **torch 2.11.0+rocm7.13.0**. `torchvision`/`torchaudio` are not
needed by this pipeline. Rollback: `deactivate; source venv/bin/activate`.

Same diagnostics afterwards:

| measurement | rocm6.2 / torch 2.5.1 | rocm7.13 / torch 2.11 | change |
|---|---|---|---|
| GEMM 4096 | 535 TFLOPS | 598 | +12% |
| GEMM 14336 | 566 TFLOPS | 676 | +19% |
| **GEMM 139336** | **53 TFLOPS** | **607** | **11.5x** |
| **SDPA default** | 48.29 ms | **6.99 ms** | **6.9x** |
| SDPA math | 49.82 ms | 38.16 ms | (now distinct — the switch works) |
| lm_head + CE | 249.6 ms | 92.1 ms | 2.7x |
| 2 layers + LoRA | 464.5 ms | 209.7 ms | 2.2x |

## 6. Liger kernel — memory, not speed

`use_liger_kernel: true` patches in Liger's fused Triton kernels; the one that
matters at this vocab size is `fused_linear_cross_entropy`, which never
materializes the `B x 2048 x 139,336` logits tensor or its fp32 copy.

| | without Liger | with Liger |
|---|---|---|
| step time | 7.90 s | **7.90 s** (unchanged) |
| `train_mem_gpu_peaked_delta` | 162,963 MB | **116,876 MB** |
| peak total | 176 GiB (96%) | **131 GiB (72%)** |
| train_loss | 2.30094 | 2.30075 |

46 GB back for free, so it is on by default. It chunks along tokens rather than
vocab, so it does **not** rescue a badly-tiled LM-head GEMM — that was fixed by
the hipBLASLt upgrade instead.

**Version constraint:** liger-kernel 0.6.0+ hard-requires transformers >= 4.52,
and this pipeline pins 4.51.3. Use `liger-kernel==0.5.10`, the last release
before that gate.

## 7. Reading the memory numbers

Set `skip_memory_metrics: false` and HF reports the real peak. It is not one
field — add three:

```
before_init_mem_gpu        16,130 MB   weights, loaded before training starts
train_mem_gpu_alloc_delta   1,436 MB   what training kept
train_mem_gpu_peaked_delta 116,876 MB  transient peak above that
                          ──────────
peak                    ≈ 134,442 MB = 131 GiB of ~183 GiB usable
```

Composition at B=8, seq 2048: ~17.6 GiB static (weights 15.1 + LoRA fp32
params/grads/Adam 2.5), the rest activations at ~**0.25 MiB per token per
layer** — a useful constant for sizing other runs, and about 45% higher than a
naive hand count of saved tensors (LoRA dropout masks, RoPE copies, norm
intermediates).

The worst case is real, not hypothetical: `group_by_length` batches the longest
sequences together and HF's length sampler deliberately puts that batch
**first**, so an OOM shows up on step 1 rather than three hours in.

## 8. Things that turned out not to be worth it

- **Padding the vocab to 139,520.** Would have fixed the 53 TFLOPS GEMM, but the
  newer hipBLASLt already does (607 TFLOPS). Not worth changing the exported
  model's `vocab_size`.
- **Bigger batch.** B=16 does not fit even with Liger. B=10 would, but the model
  already runs at ~50% of this VF's GEMM ceiling, so there are only a few
  percent left.
- **Gradient checkpointing.** Would cut activations roughly 20x and allow B=32,
  at ~33% more compute. Unnecessary now that peak sits at 72%.
- **TunableOp** (`TUNE=1 bash sft/run_sft_uc.sh`). AMD measures 6-8% on MI300X
  and it was the obvious fix for the bad LM-head GEMM — but with GEMMs now at
  600-676 TFLOPS the remaining upside is small. Still free to try; results cache
  to `sft/runs/tunableop_results.csv`.
- **`TORCH_BLAS_PREFER_HIPBLASLT=1`** is exported by the runner, but
  `torch.backends.cuda.preferred_blas_library()` already reports hipBLASLt, so
  it is a no-op belt-and-braces.

One measurement left on the table: `mem_efficient` SDPA benchmarks at **4.15 ms
vs flash's 6.83 ms**, and the default dispatch picks flash. Forcing it
(`torch.backends.cuda.enable_flash_sdp(False)`) is worth ~4% of step time.
Caveat: that benchmark used `is_causal=True` with no mask, while dynamically
padded batches make transformers build a 4D mask that flash cannot take — so
real batches may already be landing on mem_efficient.

## 9. Allocator settings

`sft/run_sft_uc.sh` sets:

```bash
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8
```

**Not** `expandable_segments:True` — this ROCm build prints
"expandable_segments not supported on this platform" and ignores it, so freed
blocks never coalesce and reserved memory ratchets upward as batch shapes vary
(dynamic padding + `group_by_length`). The GC threshold makes the allocator
release cached blocks once reserved passes 80% of VRAM. Add
`max_split_size_mb:512` if OOMs persist.

## 10. Incidental breakages fixed along the way

- **`datasets` >= 4** returns a lazy `Column` from `ds[name]` instead of a list,
  so `.with_format("numpy")["col"].sum()` raises `AttributeError`. Fixed in
  `sft/build_uc_dataset.py` by summing over the iterable, which works on both
  2.x and 4.x. Worth pinning `datasets` once an environment is settled.
- **`torch` >= 2.6** refuses to unpickle the numpy RNG state inside
  `rng_state.pth` on checkpoint resume. `sft/run_sft_uc.py` allow-lists exactly
  the numpy types HF writes there.
- The `data_collator.py:741` "Creating a tensor from a list of numpy.ndarrays is
  extremely slow" warning is **benign**. It comes from
  `DataCollatorForSeq2Seq` padding labels via numpy, prints once per dataloader
  worker (4), and costs microseconds against an 8B forward pass.

## 11. Verification checklist for a new environment

```bash
source venv-rocm7/bin/activate
python sft/diag_speed.py                       # SDPA backends must differ; GEMMs ~600 TFLOPS
python sft/run_sft_uc.py --preview 2           # template + loss mask, no GPU needed

bash sft/run_sft_uc.sh \
  --set data.max_train_samples=2000 --set train.num_train_epochs=1 \
  --set train.eval_steps=20 --set train.save_steps=20 \
  --set train.skip_memory_metrics=false \
  --set train.output_dir=sft/runs/smoke
```

Expect from the smoke run: ~7.9 s/it, `train_mem_gpu_peaked_delta` ≈ 117,000 MB,
an eval at step 20, and a real `best checkpoint: .../checkpoint-20` line. Set
`eval_steps` low enough that eval actually fires — with the production value of
300 a 31-step smoke run silently exercises none of the eval/early-stopping path
and reports `best checkpoint: None`.

## 12. Current production settings

From `sft/config.yaml`: B=8 x accum 8 (effective 64), seq 2048, bf16 LoRA
r64/alpha128 on the 7 projections with embeddings frozen, Liger on, no gradient
checkpointing, 3-epoch cap, eval + save every 300 steps, early stopping patience
3 on `eval_loss`.

3,241 steps/epoch x 7.90 s ≈ 7.1 h, plus ~14 min of eval per epoch. Budget
**~7.35 h/epoch**, ~22 h if all three epochs run, realistically ~15 h with early
stopping.
