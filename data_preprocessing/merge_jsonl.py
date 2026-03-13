#!/usr/bin/env python3
"""
JSONL Merger
Merges all .jsonl files in a folder into a single output file.
Usage: python merge_jsonl.py <input_folder> <output_file>
"""

import json
import sys
import os
from pathlib import Path


def merge_jsonl(input_folder, output_file):
    folder = Path(input_folder)

    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Folder not found: {input_folder}")
        sys.exit(1)

    jsonl_files = sorted(folder.glob("*.jsonl"))

    if not jsonl_files:
        print(f"[!] No .jsonl files found in: {input_folder}")
        sys.exit(0)

    print(f"\n{'='*55}")
    print(f"  JSONL Merger")
    print(f"{'='*55}")
    print(f"  Input folder : {folder.resolve()}")
    print(f"  Output file  : {output_file}")
    print(f"  Files found  : {len(jsonl_files)}\n")

    total_written = 0
    total_skipped = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for jsonl_file in jsonl_files:
            file_written = 0
            file_skipped = 0

            with open(jsonl_file, "r", encoding="utf-8") as in_f:
                for i, line in enumerate(in_f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json.loads(line)  # validate JSON
                        out_f.write(line + "\n")
                        file_written += 1
                    except json.JSONDecodeError as e:
                        print(f"  [!] {jsonl_file.name} line {i}: skipped — {e}")
                        file_skipped += 1

            total_written += file_written
            total_skipped += file_skipped
            status = f"{file_written} records"
            if file_skipped:
                status += f", {file_skipped} skipped"
            print(f"  ✅ {jsonl_file.name:<40} {status}")

    out_size_kb = os.path.getsize(output_file) / 1024
    print(f"\n{'─'*55}")
    print(f"  Total records written : {total_written}")
    if total_skipped:
        print(f"  Total records skipped : {total_skipped}")
    print(f"  Output size           : {out_size_kb:.1f} KB")
    print(f"  Saved to              : {output_file}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_jsonl.py <input_folder> <output_file>")
        print("\nExample:")
        print("  python merge_jsonl.py ./data merged.jsonl")
        sys.exit(1)

    merge_jsonl(sys.argv[1], sys.argv[2])