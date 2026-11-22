#!/usr/bin/env bash
# Fetch every benchmark dataset ON THE LAB BOX (user rule: Drive via ~/bin/rclone, public sets from Hugging Face;
# nothing is copied from the workstation). Rebuilds the derived sets with the original recipes and checks them
# against fingerprints of the copies used in the earlier runs. Log: ~/bench_mix/fetch.log, verdicts in CHECKS.txt.
set -uo pipefail
R=~/bin/rclone; PY=~/tts_env/bin/python
REPO=~/sinllama-continual-pretraining; TB=~/tb_eval; BM=~/bench_mix
mkdir -p $BM/data $TB/belebele $TB/rag $TB/embedders
cd $BM
CH=$BM/CHECKS.txt; : > $CH
log() { echo "[$(date '+%F %T')] $*"; }
check() { echo "$1: $2" | tee -a $CH; }

# 1. SinhalaMMLU + English MMLU (Drive)
log "MMLU"
$R copy gdrive:sinllama_cpt/SinhalaMMLU $REPO/benchmark/mmlu/SinhalaMMLU 2>&1 | grep -v NOTICE
$R check gdrive:sinllama_cpt/SinhalaMMLU $REPO/benchmark/mmlu/SinhalaMMLU --one-way 2>&1 | grep -E "differences|ERROR" | tee -a $CH
$R copy gdrive:sinllama_cpt/mmlu-english/all $REPO/benchmark/mmlu/english_mmlu/all 2>&1 | grep -v NOTICE
$R check gdrive:sinllama_cpt/mmlu-english/all $REPO/benchmark/mmlu/english_mmlu/all --one-way 2>&1 | grep -E "differences|ERROR" | tee -a $CH

# 2. Belebele (HF facebook/belebele)
log "Belebele"
$PY - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
for l in ("sin_Sinh", "eng_Latn"):
    p = hf_hub_download("facebook/belebele", f"data/{l}.jsonl", repo_type="dataset")
    shutil.copy(p, os.path.expanduser(f"~/tb_eval/belebele/{l}.jsonl"))
EOF
for l in sin_Sinh eng_Latn; do check "belebele $l rows (want 900)" "$(wc -l < $TB/belebele/$l.jsonl)"; done

# 3. Global-PIQA (HF mrlbenchmarks), sin_sinh subset -> the 0.jsonl the evaluators read
log "Global-PIQA"
$PY - <<'EOF' 2>&1 | tee -a $CH
import hashlib, os, pandas as pd
from huggingface_hub import hf_hub_download
h = lambda xs: hashlib.md5("\n".join(map(str, xs)).encode()).hexdigest()
want = {"parallel": ("44c37415a686d021db644277d0bbd1fe", "24589afaeb0ec431c0a84a5e811ae801"),
        "nonparallel": ("cba453266efa28cd5656fb805a199838", "e7543937eff604cd185110748ec3476b")}
for v in ("parallel", "nonparallel"):
    try:   # the datasets-server parquet of config sin_sinh = what the earlier runs used
        from datasets import load_dataset
        d = load_dataset(f"mrlbenchmarks/global-piqa-{v}", "sin_sinh", split="test").to_pandas()
    except Exception as e:
        print(f"piqa {v}: load_dataset failed ({str(e)[:120]}), using the tsv")
        d = pd.read_csv(hf_hub_download(f"mrlbenchmarks/global-piqa-{v}", f"data/{v}_sin_sinh.tsv",
                                        repo_type="dataset"), sep="\t")
    out = os.path.expanduser(f"~/sinllama-continual-pretraining/benchmark/mrlbenchmarks/global-piqa-{v}")
    os.makedirs(out, exist_ok=True)
    d.to_json(f"{out}/0.jsonl", orient="records", lines=True, force_ascii=False)
    got = (h(d.example_id.tolist()), h(d.label.tolist()))
    print(f"piqa {v}: {len(d)} rows, ids+labels match earlier copy: {got == want[v]}")
EOF

# 4. XL-Sum v2.0 (HF csebuetnlp/xlsum tarballs) + the seed-42 English sample (original recipe)
log "XL-Sum"
X=~/xlsum_eval; mkdir -p $X/raw
$PY - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
for l in ("sinhala", "english"):
    shutil.copy(hf_hub_download("csebuetnlp/xlsum", f"data/{l}_XLSum_v2.0.tar.bz2", repo_type="dataset"),
                os.path.expanduser(f"~/xlsum_eval/raw/{l}.tar.bz2"))
EOF
mkdir -p $X/sinhala_XLSum_v2.0 $X/raw/english
tar -xjf $X/raw/sinhala.tar.bz2 -C $X/sinhala_XLSum_v2.0 && tar -xjf $X/raw/english.tar.bz2 -C $X/raw/english
$PY - <<'EOF'
import os, random
X = os.path.expanduser("~/xlsum_eval"); out = f"{X}/english_XLSum_v2.0_sample"; os.makedirs(out, exist_ok=True)
for split, n in [("train", 3249), ("val", 500), ("test", 500)]:
    rows = open(f"{X}/raw/english/english_{split}.jsonl").readlines()
    pick = sorted(random.Random(42).sample(range(len(rows)), n))
    open(f"{out}/english_{split}.jsonl", "w").writelines(rows[i] for i in pick)
EOF
( cd $X && md5sum sinhala_XLSum_v2.0/sinhala_{train,val,test}.jsonl english_XLSum_v2.0_sample/english_{train,val,test}.jsonl ) > $BM/xlsum.md5
for f in "0b06f0113e54d742f7450aeb8398c00d sinhala_XLSum_v2.0/sinhala_train.jsonl" \
         "e5531e45c680f774852d87e4d1732332 sinhala_XLSum_v2.0/sinhala_val.jsonl" \
         "df84128ead5c6ef139a62cf6e0b18093 sinhala_XLSum_v2.0/sinhala_test.jsonl" \
         "eac1363a0fa4c47a8d8cc885e8ddd49f english_XLSum_v2.0_sample/english_train.jsonl" \
         "d3ec9d22261214c890aa299045dcefc8 english_XLSum_v2.0_sample/english_val.jsonl" \
         "b2dd44a7ac2b7ed87ac3959103073d78 english_XLSum_v2.0_sample/english_test.jsonl"; do
  grep -q "^${f%% *}  ${f#* }$" $BM/xlsum.md5 && check "xlsum ${f#* }" "identical to earlier copy" \
                                              || check "xlsum ${f#* }" "DIFFERENT md5"
done

# 5. M2DS (HF KushanH/m2ds) + the dedup train split (m2ds_dedup.py, same as the earlier runs)
log "M2DS"
$PY - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
for l in ("sinhala", "english"):
    os.makedirs(os.path.expanduser(f"~/m2ds/{l}"), exist_ok=True)
    for s in ("train", "validation", "test"):
        shutil.copy(hf_hub_download("KushanH/m2ds", f"{l}/{s}.json", repo_type="dataset"),
                    os.path.expanduser(f"~/m2ds/{l}/{s}.json"))
EOF
( cd ~/m2ds/english && md5sum train.json validation.json test.json ) | tee $BM/m2ds_en.md5
grep -q 5d0bd35bb44d6a4222a6ea68f6c3589e $BM/m2ds_en.md5 && grep -q fc350403bc424b89ba47a5b90bc8f175 $BM/m2ds_en.md5 \
  && check "m2ds english raw" "identical to earlier copy" || check "m2ds english raw" "DIFFERENT md5"
$PY $BM/m2ds_dedup.py ~/m2ds ~/m2ds_dedup > $BM/m2ds_dedup.log 2>&1
for l in sinhala english; do check "m2ds dedup $l train clusters (want si 2396 / en 10670)" "$(wc -l < ~/m2ds_dedup/$l/train.json)"; done

# 6. Textbooks for the RAG corpus (Drive) -> chunks.jsonl (embedding runs on the GPU later, in the chain)
log "textbooks"
$R copy gdrive:textbooks.tar.gz $BM/data/ 2>&1 | grep -v NOTICE
mkdir -p $BM/data/tb && tar -xzf $BM/data/textbooks.tar.gz -C $BM/data/tb
MD=$(find $BM/data/tb -type d -name markdown | head -1)
(cd $REPO && $PY $TB/rag/chunk_textbooks.py "$MD" $TB/rag/chunks.jsonl) > $BM/chunk.log 2>&1
check "rag chunks (want 33090, md5 69dd96e9b7eadb674b08273f3e66e767)" "$(wc -l < $TB/rag/chunks.jsonl) $(md5sum < $TB/rag/chunks.jsonl | cut -d' ' -f1)"
B=$(ls -d ~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/*/ | head -1); ln -sfn "$B" $TB/embedders/bge-m3

log "FETCH_DONE"; cat $CH
