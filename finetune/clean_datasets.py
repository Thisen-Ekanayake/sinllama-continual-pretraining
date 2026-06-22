"""
De-duplicate and de-contaminate the three classification datasets, in place.

Policy (test is the gold benchmark and is preserved intact):
  1. drop intra-split duplicate texts (keep first) in each of train/val/test
  2. remove from TRAIN any text that also appears in val or test
  3. remove from VAL any text that also appears in test
Matching is on a normalized text key (strip + collapse whitespace + casefold).
Row *content* is never edited -- cleaning only drops rows.

Originals are copied to <data_dir>/backup_pre_clean/ before overwriting.
"""
import os
import re
import shutil
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = {
    "writing":   ("../data/Writing",   "writing_style", "comments"),
    "sentiment": ("../data/sentiment", "sentiment",      "phrase"),
    "news":      ("../data/news",      "news",           "comments"),
}


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def find_col(df, target):
    for c in df.columns:
        if c.strip() == target:
            return c
    raise KeyError(f"column '{target}' not found in {list(df.columns)}")


def path(data_dir, stem, split):
    return os.path.join(HERE, data_dir, f"{stem}_{split}.csv")


for task, (data_dir, stem, text_col) in TASKS.items():
    print("=" * 60)
    print(f"TASK: {task}")
    dfs, cols, keys = {}, {}, {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(path(data_dir, stem, split), dtype=str, keep_default_na=False)
        col = find_col(df, text_col)
        dfs[split], cols[split] = df, col
        keys[split] = df[col].map(norm)

    before = {s: len(dfs[s]) for s in dfs}

    # 1) intra-split dedup (keep first)
    for s in ("train", "val", "test"):
        dup = keys[s].duplicated(keep="first")
        dfs[s] = dfs[s][~dup].reset_index(drop=True)
        keys[s] = keys[s][~dup].reset_index(drop=True)

    test_set = set(keys["test"])
    val_set = set(keys["val"])

    # 2) train must be disjoint from val and test
    tr_mask = keys["train"].isin(test_set | val_set)
    n_tr_removed = int(tr_mask.sum())
    dfs["train"] = dfs["train"][~tr_mask].reset_index(drop=True)

    # 3) val must be disjoint from test
    va_mask = keys["val"].isin(test_set)
    n_va_removed = int(va_mask.sum())
    dfs["val"] = dfs["val"][~va_mask].reset_index(drop=True)

    # back up + overwrite
    bdir = os.path.join(HERE, data_dir, "backup_pre_clean")
    os.makedirs(bdir, exist_ok=True)
    for s in ("train", "val", "test"):
        p = path(data_dir, stem, s)
        shutil.copy2(p, os.path.join(bdir, os.path.basename(p)))
        dfs[s].to_csv(p, index=False)

    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {before[s]:6d} -> {len(dfs[s]):6d}  "
              f"(removed {before[s] - len(dfs[s])})")
    print(f"  cross-split removed: {n_tr_removed} from train (in val/test), "
          f"{n_va_removed} from val (in test)")
    print(f"  backup -> {os.path.relpath(bdir, HERE)}/")
