#!/usr/bin/env python
"""Where is the MI300X time going? Micro-benchmarks for the SFT step.

The UltraChat run measures ~13.5 s per optimizer step (8 micro-batches of 8
dialogues), i.e. ~195 TFLOPS by HF's 6*N*tokens convention -- roughly 15% of
the MI300X's bf16 peak. This script isolates the three things that could be
eating the rest, so the fix is chosen from data instead of guesswork:

  1. raw bf16 GEMM ceiling on this box (how fast can hipBLASLt go at all)
  2. which SDPA backend actually runs -- on ROCm, PyTorch silently falls back
     to the O(L^2) math backend when the AOTriton/CK kernels are unavailable
  3. the lm_head + cross-entropy tail over the 139,336-token vocabulary
  4. a real 2-layer LlamaForCausalLM fwd+bwd, extrapolated to 32 layers,
     with and without LoRA -- predicts the step time from first principles

Run on the pod:  python sft/diag_speed.py
Nothing is written and no config is touched; delete this file when done.
"""

import time

import torch

B, S, H, KV, HD, VOCAB, LAYERS, FFN = 8, 2048, 32, 8, 128, 139336, 32, 14336
HID = H * HD
DEV = "cuda"
DT = torch.bfloat16


def timed(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def header(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main():
    print(f"torch {torch.__version__}   hip {getattr(torch.version, 'hip', None)}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    for lib in ("flash_attn", "liger_kernel"):
        try:
            m = __import__(lib)
            print(f"{lib:14s} available  {getattr(m, '__version__', '')}")
        except ImportError:
            print(f"{lib:14s} NOT installed")
    print(f"hipBLASLt preferred: {torch.backends.cuda.preferred_blas_library()}")

    # ---- 1. GEMM ceiling -------------------------------------------------
    header("1. bf16 GEMM ceiling (what the hardware will actually give us)")
    for m, k, n in [(16384, 4096, 4096), (16384, 4096, FFN), (16384, 4096, VOCAB)]:
        a = torch.randn(m, k, device=DEV, dtype=DT)
        b = torch.randn(k, n, device=DEV, dtype=DT)
        dt = timed(lambda: a @ b)
        print(f"  [{m} x {k}] @ [{k} x {n}]  {dt * 1e3:7.2f} ms  "
              f"{2 * m * k * n / dt / 1e12:7.1f} TFLOPS")
        del a, b
    torch.cuda.empty_cache()

    # ---- 2. SDPA backend -------------------------------------------------
    header("2. SDPA: which backend runs, and what each one costs")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(B, H, S, HD, device=DEV, dtype=DT, requires_grad=True)
    k = torch.randn(B, H, S, HD, device=DEV, dtype=DT, requires_grad=True)
    v = torch.randn(B, H, S, HD, device=DEV, dtype=DT, requires_grad=True)
    F = torch.nn.functional

    def fwd_bwd():
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o.sum().backward()

    base = timed(fwd_bwd, warmup=2, iters=5)
    print(f"  default dispatch          {base * 1e3:8.2f} ms  (fwd+bwd, B={B} S={S})")
    for name, backend in [("flash", SDPBackend.FLASH_ATTENTION),
                          ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
                          ("math (O(L^2))", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                dt = timed(fwd_bwd, warmup=2, iters=5)
            flag = "  <-- matches default" if abs(dt - base) / base < 0.08 else ""
            print(f"  {name:24s}  {dt * 1e3:8.2f} ms{flag}")
        except Exception as e:
            print(f"  {name:24s}  unavailable: {type(e).__name__}: {str(e)[:60]}")
    del q, k, v
    torch.cuda.empty_cache()

    # ---- 3. lm_head + cross entropy over a 139k vocab --------------------
    header("3. lm_head + cross-entropy tail (vocab 139,336)")
    tokens = B * S
    hidden = torch.randn(tokens, HID, device=DEV, dtype=DT, requires_grad=True)
    head = torch.nn.Linear(HID, VOCAB, bias=False, device=DEV, dtype=DT)
    head.weight.requires_grad_(False)  # frozen, as in the LoRA run
    labels = torch.randint(0, VOCAB, (tokens,), device=DEV)

    def head_ce():
        logits = head(hidden)
        loss = F.cross_entropy(logits.float(), labels)  # transformers upcasts
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    dt = timed(head_ce, warmup=2, iters=5)
    print(f"  standard (materialize logits + fp32 upcast)  {dt * 1e3:7.1f} ms  "
          f"peak +{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        liger = LigerFusedLinearCrossEntropyLoss()
        torch.cuda.reset_peak_memory_stats()

        def liger_ce():
            liger(head.weight, hidden, labels).backward()

        dt_l = timed(liger_ce, warmup=2, iters=5)
        print(f"  liger fused linear+CE                        {dt_l * 1e3:7.1f} ms  "
              f"peak +{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB   "
              f"({dt / dt_l:.2f}x faster)")
    except Exception as e:
        print(f"  liger fused linear+CE: unavailable ({type(e).__name__}: {str(e)[:50]})")
    del hidden, head, labels
    torch.cuda.empty_cache()

    # ---- 4. real decoder layers, extrapolated ----------------------------
    header("4. real LlamaForCausalLM fwd+bwd (2 layers -> extrapolate to 32)")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=HID, intermediate_size=FFN,
                      num_hidden_layers=2, num_attention_heads=H, num_key_value_heads=KV,
                      head_dim=HD, max_position_embeddings=8192, torch_dtype="bfloat16",
                      attn_implementation="sdpa")
    model = LlamaForCausalLM(cfg).to(DEV, DT)
    ids = torch.randint(0, VOCAB, (B, S), device=DEV)

    def run(m):
        def f():
            m(input_ids=ids, labels=ids).loss.backward()
            m.zero_grad(set_to_none=True)
        return f

    for label, m in [("frozen base + no LoRA", model)]:
        torch.cuda.reset_peak_memory_stats()
        dt = timed(run(m), warmup=2, iters=5)
        per_layer = dt / 2
        print(f"  {label:26s} {dt * 1e3:7.1f} ms for 2 layers "
              f"-> {per_layer * LAYERS * 1e3:7.0f} ms for {LAYERS} "
              f"(x8 accum = {per_layer * LAYERS * 8:.1f} s/step)")
        print(f"  {'':26s} peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
              f"for 2 layers")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        peft_model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        dt = timed(run(peft_model), warmup=2, iters=5)
        per_layer = dt / 2
        print(f"  {'+ LoRA r64 (as configured)':26s} {dt * 1e3:7.1f} ms for 2 layers "
              f"-> {per_layer * LAYERS * 1e3:7.0f} ms for {LAYERS} "
              f"(x8 accum = {per_layer * LAYERS * 8:.1f} s/step)")
    except Exception as e:
        print(f"  LoRA variant failed: {type(e).__name__}: {e}")

    print("\nCompare the last number against the 13.5 s/step the real run logs.\n"
          "A close match means the model itself is the cost (fix: bigger batch,\n"
          "liger, FA2). A large gap means overhead outside the model (dataloader,\n"
          "allocator churn, checkpoint saves).")


if __name__ == "__main__":
    main()
