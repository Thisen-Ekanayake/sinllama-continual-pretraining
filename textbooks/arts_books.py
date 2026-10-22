"""Which downloaded textbooks are in scope (arts / technical), and where each goes."""
import re
import unicodedata
from pathlib import Path

PDF_ROOT = Path(__file__).parent / "pdfs"
OUT_ROOT = Path(__file__).parent / "markdown"
GRADES = range(6, 12)

# folder-name prefix -> subject slug (prefixes are matched after whitespace normalisation)
SUBJECTS = {
    "ඉතිහාසය": "history",
    "භූගෝල විද්‍යාව": "geography",
    "පුරවැසි අධ්‍යාපනය": "civic_education",
    "සිංහල භාෂාව හා සාහිත්‍ය": "sinhala",
    "සිංහල සාහිත්‍යය සංග්‍රහය": "sinhala_literature_anthology",
    "සිංහල සාහිත්‍ය රසාස්වාදය": "sinhala_literature_appreciation",
    "සන්නිවේදනය හා මාධ්‍ය අධ්‍යයනය": "communication_media",
    "බුද්ධ ධර්මය": "buddhism",
    "කතෝලික ධර්මය": "catholicism",
    "ක්‍රිස්තියානි ධර්මය": "christianity",
    "ක්‍රිස්තු ධර්මය": "christianity",
    "ඉස්ලාම්": "islam",
    "සෞඛ්‍ය": "health_pe",
    "සෞඛය": "health_pe",
    "ව්‍යවසායකත්ව අධ්‍යයනය": "entrepreneurship",
}
TECH_SUBJECTS = {
    "ගණිතය": "mathematics",
    "විද්‍යාව": "science",
    "තොරතුරු හා සන්නිවේදන තාක්ෂණය": "ict",
    "ප්‍රායෝගික": "practical_technical_skills",
    "නිර්මාණකරණය හා ඉදිකිරීම්": "design_construction_tech",
    "නිර්මාණකරණය හා යාන්ත්‍රික": "design_mechanical_tech",
    "නිර්මාණකරණය විදුලිය": "design_electrical_electronic_tech",
    "කෘෂි හා ආහාර": "agriculture_food_tech",
    "ජලජ ජීව": "aquatic_bioresources_tech",
    "ව්‍යාපාර හා ගිණුම": "business_accounting",
    "ගෘහ ආර්ථික": "home_economics",
    "ශි්ල්ප කලා": "crafts",
}
SKIP = {(8, "health_pe"), (10, "crafts")}  # no text layer (scanned) - needs OCR later


def books(kind="arts"):
    """Yield (grade, subject, [pdf paths]) for kind 'arts' or 'technical'.
    Several books of one subject in a grade (Maths I/II/III) are yielded separately
    with a part suffix on the subject (mathematics_part_2 ...)."""
    table = SUBJECTS if kind == "arts" else TECH_SUBJECTS
    for g in GRADES:
        found = {}
        for d in sorted((PDF_ROOT / f"grade_{g}").iterdir()):
            name = unicodedata.normalize("NFC", " ".join(d.name.split()))
            subject = next((s for p, s in table.items() if name.startswith(p)), None)
            if subject and (g, subject) not in SKIP:
                found.setdefault(subject, []).append((part_number(name), d))
        for subject, dirs in found.items():
            for part, d in sorted(dirs):
                pdfs = sorted(d.glob("*.pdf"), key=chapter_order)
                yield g, subject + (f"_part_{part}" if len(dirs) > 1 else ""), pdfs


def part_number(name):
    """Book part from its folder name: 'I කොටස' / 'II කොටස' / '1 කොටස' / 'කියවීම් පොත'."""
    m = re.search(r"\b(III|II|I|[123])\s*කොටස", name)
    if m:
        return {"I": 1, "II": 2, "III": 3}.get(m.group(1)) or int(m.group(1))
    return 2 if "වැඩ පොත" in name else 1  # ICT: reader = 1, workbook = 2


def chapter_order(path):
    """Multi-PDF books (one PDF per chapter) in chapter order."""
    nums = re.findall(r"\d+", path.stem)
    return (int(nums[0]) if nums else 0, path.name)


if __name__ == "__main__":
    import sys
    for g, s, pdfs in books(sys.argv[1] if len(sys.argv) > 1 else "arts"):
        print(g, s, [p.name for p in pdfs])
