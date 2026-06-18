import json
import re
import sys

def clean_text(text):
    REMOVE_RANGES = [
        (0x0100, 0x024F),   # Latin Extended A/B
        (0x0370, 0x03FF),   # Greek
        (0x0400, 0x052F),   # Cyrillic
        (0x0530, 0x058F),   # Armenian
        (0x0590, 0x05FF),   # Hebrew
        (0x0600, 0x06FF),   # Arabic
        (0x0700, 0x074F),   # Syriac
        (0x0900, 0x097F),   # Devanagari
        (0x0980, 0x09FF),   # Bengali
        (0x0B80, 0x0BFF),   # Tamil
        (0x0C80, 0x0CFF),   # Kannada
        (0x0D00, 0x0D7F),   # Malayalam
        (0x0E00, 0x0E7F),   # Thai
        (0x1000, 0x109F),   # Myanmar
        (0x1E00, 0x1EFF),   # Latin Extended Additional
        (0x1F00, 0x1FFF),   # Greek Extended
        (0x3000, 0x303F),   # CJK Symbols and Punctuation
        (0x3040, 0x309F),   # Hiragana
        (0x3130, 0x318F),   # Hangul Compatibility Jamo
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0xAC00, 0xD7AF),   # Hangul Syllables
        (0xFE70, 0xFEFF),   # Arabic Presentation Forms-B
        (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
    ]

    def should_remove(cp):
        for start, end in REMOVE_RANGES:
            if start <= cp <= end:
                return True
        return False

    text = "".join(ch for ch in text if not should_remove(ord(ch)))

    # Keep only Sinhala, English, numerals, whitespace, punctuation
    cleaned = re.sub(
        r"[^\u0D80-\u0DFF"
        r"a-zA-Z"
        r"0-9"
        r"\s"
        r".,!?;:'\"\(\)\[\]{}\-\u2013\u2014\u2026%@#&*+=/\u200D]",  # ZWJ preserved
        "",
        text
    )
    return cleaned


def process_jsonl(input_path):
    cleaned_count = 0
    total_count = 0
    lines_out = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            total_count += 1
            record = json.loads(line)
            if "text" in record:
                original = record["text"]
                record["text"] = clean_text(original)
                if record["text"] != original:
                    cleaned_count += 1
            lines_out.append(json.dumps(record, ensure_ascii=False))

    # Overwrite the same input file
    with open(input_path, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(lines_out) + "\n")

    print(f"Done. Processed {total_count} records, cleaned {cleaned_count}.")


if len(sys.argv) == 2:
    process_jsonl(sys.argv[1])