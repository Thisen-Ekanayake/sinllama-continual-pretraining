"""Reverse-question step: show a model ONLY its own answer and ask what question produced it.

    MODEL=SinLlama_wiki_clean python gen_question.py answers_SinLlama_wiki_clean.json questions_SinLlama_wiki_clean.json

The input file holds only {id, generated_answer}; the original prompt never reaches this
machine, so it cannot leak into the model's context.
"""
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = f"/root/sinllama-continual-pretraining/models/{os.environ['MODEL']}"
src, dst = sys.argv[1], sys.argv[2]
BATCH = int(os.environ.get("BATCH", 8))

# "The following is an answer given to some question or instruction. ...
#  What was the original question or instruction asked to get the above answer? Write only that question, in one sentence."
INSTRUCTION_HEAD = "පහත දැක්වෙන්නේ යම් ප්‍රශ්නයකට හෝ උපදෙසකට ලබා දුන් පිළිතුරකි."
INSTRUCTION_TAIL = ("ඉහත පිළිතුර ලබා ගැනීමට අසන ලද මුල් ප්‍රශ්නය හෝ උපදෙස කුමක්ද? "
                    "එම ප්‍රශ්නය පමණක් එක් වාක්‍යයකින් ලියන්න.")


def render(answer):
    # same UltraChat template the three chat models were trained on
    user = f"{INSTRUCTION_HEAD}\n\nපිළිතුර:\n{answer.strip()}\n\n{INSTRUCTION_TAIL}"
    return f"### User:\n{user}\n\n### Assistant:\n"


rows = json.load(open(src, encoding="utf-8"))
assert all(set(r) == {"id", "generated_answer"} for r in rows), "input must carry only id + generated_answer"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
tok.pad_token = tok.pad_token or tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa", device_map="cuda").eval()
eos = tok.convert_tokens_to_ids("<|end_of_text|>")

out = []
# longest first so each batch pads little
order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["generated_answer"]))
t0 = time.time()
for b in range(0, len(order), BATCH):
    idx = order[b:b + BATCH]
    enc = tok([render(rows[i]["generated_answer"]) for i in idx], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=128, do_sample=False, repetition_penalty=1.1,
                             no_repeat_ngram_size=3, eos_token_id=eos, pad_token_id=tok.pad_token_id)
    for j, i in enumerate(idx):
        new = gen[j][enc["input_ids"].shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        text = text.split("### User:")[0].split("### Assistant:")[0].strip()
        out.append({"id": rows[i]["id"], "generated_question": text,
                    "input_tokens": int(enc["attention_mask"][j].sum()),
                    "hit_eos": bool((new == eos).any())})
    print(f"{os.environ['MODEL']}: {len(out)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)

out.sort(key=lambda r: int(r["id"]))
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done", dst)
