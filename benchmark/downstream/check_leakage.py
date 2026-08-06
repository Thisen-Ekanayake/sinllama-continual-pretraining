"""
Train/val/test leakage check for the three downstream datasets.

For each task, reports overlap of the *text* column between splits, using:
  - exact match (after stripping whitespace)
  - normalized match (strip + collapse whitespace + casefold)
and flags any test rows whose text also appears in train.
"""
import os
import re
import pandas as pd

TASKS = {
    "writing":   ("../data/Writing",   "writing_style", "comments"),
    "sentiment": ("../data/sentiment", "sentiment",      "phrase"),
    "news":      ("../data/news",      "news",           "comments"),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def load(data_dir, stem, split):
    df = pd.read_csv(os.path.join(HERE, data_dir, f"{stem}_{split}.csv"))
    df.columns = [c.strip() for c in df.columns]
    return df


for task, (data_dir, stem, text_col) in TASKS.items():
    print("=" * 64)
    print(f"TASK: {task}   (text column: {text_col})")
    print("=" * 64)
    try:
        tr = load(data_dir, stem, "train")
        va = load(data_dir, stem, "val")
        te = load(data_dir, stem, "test")
    except Exception as e:
        print(f"  could not load splits: {e}\n")
        continue

    if text_col not in tr.columns:
        print(f"  !! text column '{text_col}' not found; columns = {list(tr.columns)}\n")
        continue

    tr_txt = tr[text_col].astype(str)
    te_txt = te[text_col].astype(str)
    va_txt = va[text_col].astype(str)

    print(f"  sizes: train={len(tr)}  val={len(va)}  test={len(te)}")

    # within-split exact dupes
    print(f"  intra-train exact dupes: {tr_txt.duplicated().sum()}")
    print(f"  intra-test  exact dupes: {te_txt.duplicated().sum()}")

    for name, exact in [("EXACT (strip)", lambda s: s.str.strip()),
                        ("NORMALIZED", lambda s: s.map(norm))]:
        tr_set = set(exact(tr_txt))
        va_set = set(exact(va_txt))
        te_keys = exact(te_txt)
        va_keys = exact(va_txt)

        test_in_train = te_keys.isin(tr_set)
        val_in_train = va_keys.isin(tr_set)
        test_in_val = te_keys.isin(va_set)

        n = len(te_keys)
        print(f"  [{name}]")
        print(f"     test rows also in train : {test_in_train.sum()} / {n} "
              f"({100*test_in_train.mean():.2f}%)")
        print(f"     val  rows also in train : {val_in_train.sum()} / {len(va_keys)} "
              f"({100*val_in_train.mean():.2f}%)")
        print(f"     test rows also in val   : {test_in_val.sum()} / {n} "
              f"({100*test_in_val.mean():.2f}%)")
    print()
