#!/bin/bash
# Last step of the M2DS benchmark, on the local machine. First copy the lab box's
# results into this run directory:
#   scp -r cse_g2@10.8.100.21:~/m2ds_eval/results/* results/data/m2ds_lab_4070ti_20261031/
# then run this script. It adds the Sinhala BERTScore (local /ml/HelaBERT_Large,
# never uploaded to the server) and builds results/latex/M2DS_Summarization.{tex,pdf},
# copying the PDF to results/pdf/. Uses the GPU, or the CPU if that fails (busy GPU).
set -eu
cd "$(dirname "$0")"
RUN=..
[ -f $RUN/sinhala/rouge.json ] && [ -f $RUN/english/rouge.json ] || { echo "results not copied into $RUN yet"; exit 1; }
bs() {  # bs <preds_dir> <out_prefix>
  DEVICE=${DEVICE:-cuda} python3 bertscore_m2ds.py $1 $2 sinhala > $2.txt 2>&1 \
    || { echo "GPU run failed, BERTScore on CPU (slower)"; DEVICE=cpu python3 bertscore_m2ds.py $1 $2 sinhala > $2.txt 2>&1; }
  grep -v -i warn $2.txt
}
bs $RUN/sinhala/preds $RUN/sinhala/bertscore_helabert
# reference row: v01 fine-tuned on the official (leaky) train split
[ -d $RUN/sinhala/preds_official ] && bs $RUN/sinhala/preds_official $RUN/sinhala/bertscore_helabert_official
python3 make_m2ds_tex.py
cd $RUN/../../latex
for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error M2DS_Summarization.tex > /dev/null; done
mkdir -p ../pdf && cp M2DS_Summarization.pdf ../pdf/
echo "wrote $(pwd)/M2DS_Summarization.pdf and ../pdf/M2DS_Summarization.pdf"
