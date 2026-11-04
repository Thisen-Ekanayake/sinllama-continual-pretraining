"""Sinhala XL-Sum (v2.0 test, 500 articles) summary generation on the 16 GB lab box.

Raw 1-shot prompt (same for every model, no chat template), greedy decoding,
stop at the first newline. Articles are truncated to their first MAX_WORDS
whitespace words so every model sees the same text (llama-3-8b spends ~12
tokens per Sinhala word and must fit in 8192). bf16, decoder on GPU,
embed_tokens + lm_head on CPU (lm_head only ever sees the last position).
Resumable: appends one JSON line per article to OUT.
Usage: python xlsum_gen.py MODEL_DIR OUT_JSONL
"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

DATA = os.path.expanduser("~/xlsum_eval/sinhala_XLSum_v2.0")
EXEMPLAR_ID = "140312_ukraine_west_russia"      # train split, 118 words
MAX_WORDS = 400                                 # 397/500 test articles fit whole
MAX_SUMMARY_WORDS = 60                          # longest test reference: 47 words
PROMPT = ("පහත පුවත් ලිපිය එක් වාක්‍යයකින් සාරාංශ කරන්න.\n\n"
          "ලිපිය: {ex_text}\nසාරාංශය: {ex_summary}\n\n"
          "ලිපිය: {text}\nසාරාංශය:")


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="sdpa")
    emb = model.model.embed_tokens
    model.model.embed_tokens = None                 # keep it on CPU through .to()
    model.model.to("cuda")
    model.model.embed_tokens = emb
    emb.register_forward_pre_hook(lambda m, a: (a[0].to("cpu"),))
    emb.register_forward_hook(lambda m, a, out: out.to("cuda"))

    def head_in(m, a):
        assert a[0].shape[1] == 1, f"lm_head got {a[0].shape[1]} positions"
        return (a[0].to("cpu"),)
    model.lm_head.register_forward_pre_hook(head_in)
    model.eval()
    print(f"GPU after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return model, tok


class StopAtNewline(StoppingCriteria):
    def __init__(self, tok, start):
        self.tok, self.start = tok, start

    def __call__(self, ids, scores, **kw):
        return "\n" in self.tok.decode(ids[0, self.start:], skip_special_tokens=True)


def main(model_dir, out):
    rows = [json.loads(l) for l in open(f"{DATA}/sinhala_test.jsonl")][:int(os.environ.get("LIMIT", 500))]
    ex = next(json.loads(l) for l in open(f"{DATA}/sinhala_train.jsonl")
              if json.loads(l)["id"] == EXEMPLAR_ID)
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    model, tok = load(model_dir)
    # New-token budget in this tokenizer's units: ~MAX_SUMMARY_WORDS words of Sinhala.
    per_word = len(tok(ex["summary"], add_special_tokens=False).input_ids) / len(ex["summary"].split())
    max_new = int(per_word * MAX_SUMMARY_WORDS) + 8
    print(f"{len(done)} done, {len(rows) - len(done)} to go; {per_word:.2f} tok/word -> max_new_tokens {max_new}", flush=True)
    t0 = time.time()
    with open(out, "a") as f:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            words = r["text"].split()
            for n_words in (MAX_WORDS, 300, 200):
                prompt = PROMPT.format(ex_text=ex["text"], ex_summary=ex["summary"],
                                       text=" ".join(words[:n_words]))
                enc = tok(prompt, return_tensors="pt").to("cuda")
                n_in = enc.input_ids.shape[1]
                oom = False
                try:
                    with torch.no_grad():
                        gen = model.generate(
                            **enc, max_new_tokens=max_new, do_sample=False,
                            num_beams=1, temperature=None, top_p=None,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id,
                            stopping_criteria=[StopAtNewline(tok, n_in)])
                except torch.cuda.OutOfMemoryError:
                    oom = True
                if not oom:
                    break
                torch.cuda.empty_cache()
                print(f"  OOM at {n_in} tokens ({n_words} words), retrying shorter", flush=True)
            raw = tok.decode(gen[0, n_in:], skip_special_tokens=True)
            pred = raw.strip().split("\n")[0].strip()
            f.write(json.dumps({
                "id": r["id"], "prediction": pred, "reference": r["summary"],
                "raw": raw, "prompt_tokens": n_in, "new_tokens": gen.shape[1] - n_in,
                "article_words_used": min(n_words, len(words)), "article_words": len(words),
            }, ensure_ascii=False) + "\n")
            f.flush()
            k = i + 1
            if k % 10 == 0 or k == len(rows):
                el = time.time() - t0
                print(f"[{k}/{len(rows)}] {el / 60:.1f} min, prompt {n_in} tok, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB | {pred[:80]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
