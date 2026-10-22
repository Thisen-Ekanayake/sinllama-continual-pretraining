"""Re-typeset one textbook page with a local vision model (llama-server, OpenAI API).

The model sees the page image AND the exact text recovered from the PDF's text
layer. Small VLMs misread Sinhala script, so the Sinhala must come from the text
layer; the image is only used to restore layout: equations (LaTeX), fractions,
tables, lists and reading order.
"""
import base64
import json
import subprocess
import tempfile
import time
import urllib.request

URL = "http://localhost:8091/v1/chat/completions"

PROMPT = """You are converting one page of a Sri Lankan school textbook (Sinhala medium) into Markdown.

You get:
1. the page image
2. TEXT: the exact text of the page, recovered from the PDF in a rough reading order. Its Sinhala spelling is CORRECT. Superscripts are marked ^{..} and subscripts _{..}.

Write the page as Markdown:
- Copy Sinhala words from TEXT exactly. Never retype Sinhala from the image and never translate.
- Use the image only to fix structure: reading order, headings (##), numbered/bulleted lists, tables (Markdown tables).
- Write every mathematical expression, equation, fraction, root, power, chemical formula and unit in LaTeX: inline $...$, display $$...$$. Fractions as \\frac{a}{b}, roots as \\sqrt{..}.
- For a diagram, graph or picture write one line: [රූපය: <its caption or a few words from its labels>]. Do not describe it further.
- Drop page numbers and running headers/footers.
- Output only the Markdown, nothing else.

TEXT:
"""


def render_png(pdf, page_number, dpi=110):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", "-f", str(page_number),
                        "-l", str(page_number), str(pdf), f"{d}/p"], check=True)
        return open(f"{d}/p.png", "rb").read()


def vlm_page(pdf, page_number, text, max_tokens=4000, retries=3):
    img = base64.b64encode(render_png(pdf, page_number)).decode()
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            {"type": "text", "text": PROMPT + text}]}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.load(r)
            msg = out["choices"][0]["message"]["content"].strip()
            if msg.startswith("```"):
                msg = msg.strip("`").removeprefix("markdown").strip()
            return msg, out["choices"][0].get("finish_reason")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(10)
