import json
import re
import sys

def clean_text(text):
    """
    Keep:
    - Sinhala Unicode block: U+0D80–U+0DFF
    - English letters: a-zA-Z
    - Numerals: 0-9
    - Punctuation & common symbols: .,!?;:'"()[]{}/-–—…%@#&*+= and whitespace
    Remove:
    - Everything else (Tamil, Hindi/Devanagari, Arabic, Chinese, etc.)
    """
    cleaned = re.sub(
        r"[^\u0D80-\u0DFF"   # Sinhala
        r"a-zA-Z"             # English
        r"0-9"                # Numerals
        r"\s"                 # Whitespace
        r".,!?;:'\"\(\)\[\]{}\-–—…%@#&*+=/"  # Punctuation
        r"]",
        "",
        text
    )
    return cleaned

def process_jsonl(input_path, output_path):
    cleaned_count = 0
    total_count = 0

    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            total_count += 1
            record = json.loads(line)

            # Clean the "text" field if it exists
            if "text" in record:
                original = record["text"]
                record["text"] = clean_text(original)
                if record["text"] != original:
                    cleaned_count += 1

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Processed {total_count} records, cleaned {cleaned_count}.")

# --- CLI usage ---
if len(sys.argv) == 3:
    process_jsonl(sys.argv[1], sys.argv[2])
elif len(sys.argv) == 2:
    process_jsonl(sys.argv[1], "output.jsonl")
else:
    print("Usage: python clean_jsonl.py input.jsonl [output.jsonl]")
    print("(Running in test mode only)")