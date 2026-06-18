import json
import sys
import unicodedata
from collections import defaultdict

# Unicode block ranges mapped to language/script names
UNICODE_BLOCKS = [
    (0x0000, 0x007F, "English / Basic Latin"),
    (0x0080, 0x00FF, "Latin Extended (Western European)"),
    (0x0100, 0x024F, "Latin Extended A/B (Central/Eastern European)"),
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic (Russian, Bulgarian, etc.)"),
    (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic Supplement"),
    (0x0900, 0x097F, "Devanagari (Hindi, Sanskrit, Marathi, Nepali)"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi (Punjabi)"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia (Oriya)"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar (Burmese)"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul Jamo (Korean)"),
    (0x1C00, 0x1C4F, "Lepcha"),
    (0x1C50, 0x1C7F, "Ol Chiki (Santali)"),
    (0x1E00, 0x1EFF, "Latin Extended Additional"),
    (0x1F00, 0x1FFF, "Greek Extended"),
    (0x2000, 0x206F, "General Punctuation"),
    (0x2070, 0x209F, "Superscripts and Subscripts"),
    (0x20A0, 0x20CF, "Currency Symbols"),
    (0x2100, 0x214F, "Letterlike Symbols"),
    (0x2150, 0x218F, "Number Forms"),
    (0x2190, 0x21FF, "Arrows"),
    (0x2200, 0x22FF, "Mathematical Operators"),
    (0x2C00, 0x2C5F, "Glagolitic"),
    (0x2C60, 0x2C7F, "Latin Extended-C"),
    (0x2C80, 0x2CFF, "Coptic"),
    (0x2D00, 0x2D2F, "Georgian Supplement"),
    (0x3000, 0x303F, "CJK Symbols and Punctuation"),
    (0x3040, 0x309F, "Hiragana (Japanese)"),
    (0x30A0, 0x30FF, "Katakana (Japanese)"),
    (0x3100, 0x312F, "Bopomofo (Chinese)"),
    (0x3130, 0x318F, "Hangul Compatibility Jamo (Korean)"),
    (0x3190, 0x319F, "Kanbun (Japanese)"),
    (0x31F0, 0x31FF, "Katakana Phonetic Extensions"),
    (0x3400, 0x4DBF, "CJK Unified Ideographs Extension A (Chinese/Japanese/Korean)"),
    (0x4E00, 0x9FFF, "CJK Unified Ideographs (Chinese/Japanese/Korean)"),
    (0xA000, 0xA48F, "Yi Syllables"),
    (0xA490, 0xA4CF, "Yi Radicals"),
    (0xA960, 0xA97F, "Hangul Jamo Extended-A (Korean)"),
    (0xAC00, 0xD7AF, "Hangul Syllables (Korean)"),
    (0xF900, 0xFAFF, "CJK Compatibility Ideographs"),
    (0xFB00, 0xFB4F, "Alphabetic Presentation Forms (Hebrew/Latin)"),
    (0xFB50, 0xFDFF, "Arabic Presentation Forms-A"),
    (0xFE30, 0xFE4F, "CJK Compatibility Forms"),
    (0xFE70, 0xFEFF, "Arabic Presentation Forms-B"),
    (0xFF00, 0xFFEF, "Halfwidth and Fullwidth Forms (CJK)"),
    (0x10000, 0x1007F, "Linear B Syllabary"),
    (0x10300, 0x1032F, "Old Italic"),
    (0x10330, 0x1034F, "Gothic"),
    (0x10400, 0x1044F, "Deseret"),
    (0x10450, 0x1047F, "Shavian"),
    (0x10480, 0x104AF, "Osmanya"),
    (0x1D000, 0x1D0FF, "Byzantine Musical Symbols"),
    (0x1D400, 0x1D7FF, "Mathematical Alphanumeric Symbols"),
    (0x1F300, 0x1F5FF, "Miscellaneous Symbols and Pictographs (Emoji)"),
    (0x1F600, 0x1F64F, "Emoticons (Emoji)"),
    (0x1F650, 0x1F67F, "Ornamental Dingbats"),
    (0x1F680, 0x1F6FF, "Transport and Map Symbols (Emoji)"),
    (0x1F900, 0x1F9FF, "Supplemental Symbols and Pictographs (Emoji)"),
    (0x20000, 0x2A6DF, "CJK Unified Ideographs Extension B"),
]

# Characters to ignore (not language-specific)
IGNORE_BLOCKS = {
    "General Punctuation",
    "Currency Symbols",
    "Letterlike Symbols",
    "Number Forms",
    "Arrows",
    "Mathematical Operators",
    "Superscripts and Subscripts",
    "Mathematical Alphanumeric Symbols",
}

def get_script(codepoint):
    for start, end, name in UNICODE_BLOCKS:
        if start <= codepoint <= end:
            return name
    return None

def analyze_text(text):
    scripts = defaultdict(int)
    for char in text:
        cp = ord(char)
        # Skip ASCII control chars, whitespace, digits, basic punctuation
        if cp < 0x0021:
            continue
        if 0x0021 <= cp <= 0x002F:  # !"#$%&'()*+,-./
            continue
        if 0x0030 <= cp <= 0x0039:  # 0-9
            continue
        if 0x003A <= cp <= 0x0040:  # :;<=>?@
            continue
        if 0x005B <= cp <= 0x0060:  # [\]^_`
            continue
        if 0x007B <= cp <= 0x007E:  # {|}~
            continue

        script = get_script(cp)
        if script and script not in IGNORE_BLOCKS:
            scripts[script] += 1

    return scripts

def process_jsonl(input_path):
    total_records = 0
    global_scripts = defaultdict(int)   # total char count per script
    record_scripts = defaultdict(int)   # how many records contain each script

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [!] Skipping invalid JSON on line {total_records}")
                continue

            text = record.get("text", "")
            if not text:
                continue

            scripts_in_record = analyze_text(text)
            for script, count in scripts_in_record.items():
                global_scripts[script] += count

            for script in scripts_in_record:
                record_scripts[script] += 1

    return total_records, global_scripts, record_scripts

def print_report(input_path, total_records, global_scripts, record_scripts):
    print("=" * 65)
    print(f"  Language / Script Detection Report")
    print(f"  File   : {input_path}")
    print(f"  Records: {total_records:,}")
    print("=" * 65)

    if not global_scripts:
        print("  No non-ASCII script characters found.")
        return

    # Sort by total character count descending
    sorted_scripts = sorted(global_scripts.items(), key=lambda x: x[1], reverse=True)

    total_chars = sum(global_scripts.values())

    print(f"  {'Script / Language':<45} {'Chars':>8}  {'%':>6}  {'Records':>8}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*6}  {'-'*8}")

    for script, count in sorted_scripts:
        pct = (count / total_chars) * 100
        recs = record_scripts[script]
        print(f"  {script:<45} {count:>8,}  {pct:>5.1f}%  {recs:>8,}")

    print("=" * 65)
    print(f"  Total script characters: {total_chars:,}")
    print("=" * 65)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_languages.py <input.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    total_records, global_scripts, record_scripts = process_jsonl(input_path)
    print_report(input_path, total_records, global_scripts, record_scripts)