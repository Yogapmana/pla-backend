"""Robust PDF → text extraction.

Many scanned/generated PDFs (i.e. some Gunadarma RPS) embed fonts without a
ToUnicode CMap: PyMuPDF then extracts only whitespace/digits, while Poppler's
``pdftotext -layout`` recovers the full text. Strategy:

1. Try ``pdftotext -layout`` (if installed — we ship ``poppler-utils`` in the
   Docker image).  ``-layout`` keeps table columns/rows intact, which matters
   for RPS "Minggu | Bahan Kajian | Sub-CPMK" tables.
2. Fall back to PyMuPDF if the binary is unavailable or returns nothing.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)

_POPPLER_LAYOUT_OPTION = "-layout"


def _extract_with_pdftotext(content: bytes) -> str | None:
    """Return pdftotext -layout output, or None if unavailable/empty."""
    binary = shutil.which("pdftotext")
    if not binary:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            pdf_path = tmp.name
        try:
            proc = subprocess.run(
                [binary, _POPPLER_LAYOUT_OPTION, pdf_path, "-"],
                capture_output=True,
                timeout=120,
            )
        finally:
            try:
                Path(pdf_path).unlink(missing_ok=True)
            except Exception:
                pass
        if proc.returncode != 0:
            logger.warning("[PDF-EXTRACT] pdftotext failed: %s", proc.stderr.decode(errors="ignore")[:200])
            return None
        text = proc.stdout.decode("utf-8", errors="ignore")
        return text.strip() or None
    except Exception as exc:
        logger.warning("[PDF-EXTRACT] pdftotext error: %s", exc)
        return None


def _extract_with_pymupdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    parts = []
    try:
        for page in doc:
            parts.append(page.get_text("text"))
    finally:
        doc.close()
    return "\n\n".join(p for p in parts if p.strip()).strip()


def extract_pdf_text(content: bytes) -> str:
    """Best-effort PDF text extraction.

    Returns empty string only when both methods fail.
    """
    text = _extract_with_pdftotext(content)
    if text:
        return text
    return _extract_with_pymupdf(content)


def extract_rps_material_section(rps_text: str, max_chars: int = 24000) -> str:
    """Trim an RPS to its teaching-plan section only.

    RPS files carry lots of non-teaching content (cover, "rancangan tugas",
    grading schemes) that only bloats the planner prompt and distracts the
    LLM. The weekly plan table (Minggu → Bahan Kajian/Sub-CPMK) is the part
    that matters. Slices from the first occurrence of the table markers and
    caps the length. Falls back to the full text if those markers are absent.
    """
    start: int | None = None
    for marker in ("Minggu", "Bahan Kajian", "Sub-CPMK"):
        i = rps_text.find(marker)
        if i >= 0:
            start = i if start is None else min(start, i)
    if start is None:
        return rps_text
    return rps_text[start : start + max_chars]