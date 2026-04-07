import os
import re
import uuid
from pathlib import Path
import pdfplumber
from docx import Document
from docx.shared import Pt, RGBColor

# Optional OCR imports (only used if available)
try:
    import pytesseract
    from pdf2image import convert_from_path
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False


_EMBED = None
_EMBED_UTIL = None
_EMBED_ATTEMPTED = False


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _get_embedder():
    global _EMBED, _EMBED_UTIL, _EMBED_ATTEMPTED

    if _EMBED_ATTEMPTED:
        return _EMBED, _EMBED_UTIL

    _EMBED_ATTEMPTED = True

    if not _env_truthy("ENABLE_SEMANTIC_SCORE"):
        return None, None

    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception as e:
        print("Semantic model disabled:", e)
        return None, None

    model_name = os.getenv("SEMANTIC_MODEL", "all-MiniLM-L6-v2").strip() or "all-MiniLM-L6-v2"
    allow_download = _env_truthy("ALLOW_MODEL_DOWNLOAD")

    orig_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    orig_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _EMBED = SentenceTransformer(model_name)
        _EMBED_UTIL = util
        return _EMBED, _EMBED_UTIL
    except Exception as e_offline:
        if not allow_download:
            print("Semantic model disabled:", e_offline)
            return None, None

        try:
            _restore_env("HF_HUB_OFFLINE", orig_hf_offline)
            _restore_env("TRANSFORMERS_OFFLINE", orig_tf_offline)
            _EMBED = SentenceTransformer(model_name)
            _EMBED_UTIL = util
            return _EMBED, _EMBED_UTIL
        except Exception as e_online:
            print("Semantic model disabled:", e_online)
            return None, None
    finally:
        _restore_env("HF_HUB_OFFLINE", orig_hf_offline)
        _restore_env("TRANSFORMERS_OFFLINE", orig_tf_offline)




def extract_text(path: Path) -> str:
    try:
        ext = path.suffix.lower()

        if ext == ".txt":
            return path.read_text(errors="ignore")

        if ext == ".pdf":
            text = ""
            with pdfplumber.open(path) as pdf:
                for p in pdf.pages:
                    t = p.extract_text()
                    if t:
                        text += t + "\n"
            if text.strip():
                return text
            # fallback to OCR if no text extracted and OCR libs available
            if _HAS_OCR:
                ocr_chunks = []
                try:
                    for img in convert_from_path(str(path)):
                        ocr_chunks.append(pytesseract.image_to_string(img))
                    return "\n".join(ocr_chunks)
                except Exception:
                    pass
            return text

        if ext == ".docx":
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)

        return ""
    except Exception:
        return ""


def _clean_cell(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _table_rows_from_pdf(path: Path) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables() or []
            for table in page_tables:
                rows: list[list[str]] = []
                for row in table or []:
                    cleaned = [_clean_cell(cell) for cell in (row or [])]
                    if any(cleaned):
                        rows.append(cleaned)
                if rows:
                    tables.append(rows)
    return tables


def _table_rows_from_docx(path: Path) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    doc = Document(path)
    for table in doc.tables:
        rows: list[list[str]] = []
        for row in table.rows:
            cleaned = [_clean_cell(cell.text) for cell in row.cells]
            if any(cleaned):
                rows.append(cleaned)
        if rows:
            tables.append(rows)
    return tables


def extract_tables(path: Path) -> list[list[list[str]]]:
    try:
        ext = path.suffix.lower()
        if ext == ".pdf":
            return _table_rows_from_pdf(path)
        if ext == ".docx":
            return _table_rows_from_docx(path)
        return []
    except Exception:
        return []


def tables_to_text(tables: list[list[list[str]]]) -> str:
    lines: list[str] = []
    for table in tables:
        for row in table:
            cleaned = [c for c in row if c]
            if cleaned:
                lines.append(" | ".join(cleaned))
    return "\n".join(lines)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()





def extract_bullets(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    bullets = []

    for l in lines:
        if l.startswith(("-", "*")):
            bullets.append(l[1:].strip())
        elif len(l) < 180 and re.search(
            r"(developed|built|created|managed|designed|led|improved|worked)",
            l,
            re.I
        ):
            bullets.append(l)

    if not bullets:
        parts = re.split(r"[.!?]", text)
        for p in parts:
            if 30 < len(p) < 250:
                bullets.append(p.strip())

    return bullets


def extract_keywords(text, top_n=20):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    stopwords = {"the", "and", "for", "with", "this", "that", "were", "your"}
    freq = {}

    for w in words:
        if len(w) > 2 and w not in stopwords:
            freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in sorted_words[:top_n]]


def keyword_score(resume, job_keywords):
    found, missing = [], []
    resume_lower = resume.lower()

    for kw in job_keywords:
        if kw.lower() in resume_lower:
            found.append(kw)
        else:
            missing.append(kw)

    score = len(found) / max(1, len(job_keywords))
    return score, found, missing


def formatting_score(text):
    score = 0
    tl = text.lower()

    if "experience" in tl:
        score += 0.3
    if "education" in tl:
        score += 0.3
    if "skills" in tl:
        score += 0.3
    if len(text) < 5000:
        score += 0.1

    return min(score, 1)


def semantic_score(resume, job) -> float | None:
    embed, util = _get_embedder()
    if embed is None:
        return None
    try:
        r = embed.encode(resume, convert_to_tensor=True)
        j = embed.encode(job, convert_to_tensor=True)
        score = float(util.cos_sim(r, j).item())
        if score < 0.0:
            return 0.0
        if score > 1.0:
            return 1.0
        return score
    except Exception:
        return None



def _apply_layout(doc: Document, layout: str) -> None:
    layout = (layout or "classic").lower()
    if layout == "modern":
        base_font = ("Arial", Pt(11), RGBColor(15, 23, 42))
        accent = RGBColor(29, 78, 216)
    elif layout == "minimal":
        base_font = ("Times New Roman", Pt(11), RGBColor(33, 37, 41))
        accent = RGBColor(90, 90, 90)
    else:  # classic
        base_font = ("Calibri", Pt(11), RGBColor(31, 41, 55))
        accent = RGBColor(37, 99, 235)

    # Normal text
    normal = doc.styles["Normal"].font
    normal.name = base_font[0]
    normal.size = base_font[1]
    normal.color.rgb = base_font[2]

    for heading in ("Heading 1", "Heading 2", "Heading 3"):
        h_style = doc.styles[heading].font
        h_style.name = base_font[0]
        h_style.size = Pt(16 if heading == "Heading 1" else 13)
        h_style.color.rgb = accent


def export_docx(name, skills, bullets, export_dir, original_text: str | None = None, layout: str = "classic"):
    try:
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        skills_list = [str(s).strip() for s in (skills or []) if str(s).strip()]
        bullets_list = [str(b).strip() for b in (bullets or []) if str(b).strip()]

        doc = Document()
        _apply_layout(doc, layout)
        doc.add_heading(name or "Candidate", level=0)

        if original_text:
            doc.add_heading("Original Resume Excerpt", level=1)
            # keep it concise
            snippet = original_text[:3000]
            doc.add_paragraph(snippet if snippet else "N/A")

        doc.add_heading("Skills", level=1)
        doc.add_paragraph(", ".join(skills_list) if skills_list else "N/A")

        doc.add_heading("Experience", level=1)
        if bullets_list:
            for b in bullets_list:
                try:
                    doc.add_paragraph(b, style="List Bullet")
                except Exception:
                    doc.add_paragraph(f"- {b}")
        else:
            doc.add_paragraph("No bullet points available.")

        doc.add_heading("AI-Improved Points", level=1)
        if bullets_list:
            for idx, b in enumerate(bullets_list, start=1):
                doc.add_paragraph(f"{idx}. {b}")
        else:
            doc.add_paragraph("No improved points available.")

        out = export_dir / f"{uuid.uuid4().hex}.docx"
        doc.save(str(out))
        return out
    except Exception as e:
        print("DOCX export failed:", e)
        return None
