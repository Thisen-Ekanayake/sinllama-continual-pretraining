import os
import json
import time
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = "data/oasst2_selected.jsonl"
OUTPUT_FILE = "data/oasst2_selected_si.jsonl"

API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_TRANSLATE_API_KEY not found in .env")


def translate_text(text, api_key):
    """
    Translate English text to Sinhala using Google Cloud Translation API
    """
    url = "https://translation.googleapis.com/language/translate/v2"

    data = {
        "q": text,
        "source": "en",
        "target": "si",
        "format": "text",
        "key": api_key,
    }

    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        payload = json.loads(body)

    try:
        return payload["data"]["translations"][0]["translatedText"]
    except Exception:
        raise RuntimeError(f"Unexpected translation response: {payload}")


def count_existing_lines(filepath):
    """
    Count how many lines already exist in the output file.
    """
    if not os.path.exists(filepath):
        return 0

    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    # Determine resume point
    processed_lines = count_existing_lines(OUTPUT_FILE)
    print(f"Resuming from line: {processed_lines}")

    total = processed_lines
    current_line = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:

        for line in f_in:
            if current_line < processed_lines:
                current_line += 1
                continue

            obj = json.loads(line)

            # Translate each message content
            for msg in obj["messages"]:
                original_text = msg["content"]

                try:
                    translated = translate_text(original_text, API_KEY)
                    msg["content"] = translated
                    time.sleep(0.1)  # prevent rate spikes
                except Exception as e:
                    print(f"Translation error: {e}")
                    print("Keeping original text.")
                    msg["content"] = original_text

            json.dump(obj, f_out, ensure_ascii=False)
            f_out.write("\n")

            total += 1
            current_line += 1

            if total % 50 == 0:
                print(f"Processed {total} conversations")

    print(f"Finished. Total translated conversations: {total}")


if __name__ == "__main__":
    main()