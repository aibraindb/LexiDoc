#!/usr/bin/env python3
"""
LexiGraph OCR/Text Extractor — single-file main

Backends (auto-detected, configurable):
  - pymupdf_text  (DEFAULT)  -> uses PyMuPDF to read digital text + bboxes
  - easyocr                  -> uses easyocr for OCR on page images
  - paddleocr                -> uses paddleocr for OCR on page images
  - pdfminer                 -> pdfminer.six text (no word bboxes), lines only

Pick backend with:
  CLI:    --backend <pymupdf_text|easyocr|paddleocr|pdfminer>
  or env: LEXI_OCR_BACKEND=<...>

Outputs JSON with:
{
  "doc_id": "<basename>",
  "backend": "<chosen>",
  "pages": [
    {
      "page_index": 0,
      "width": <px>,
      "height": <px>,
      "image_bytes": "<base64-png>",   # only when --embed-images
      "lines": [
        {"id": 0, "text": "...", "bbox": [x0,y0,x1,y1], "conf": 1.0}
      ],
      "words": [
        {"id": 0, "text": "Total", "bbox": [x0,y0,x1,y1], "conf": 1.0}
      ]
    },
    ...
  ]
}
"""
from __future__ import annotations
import argparse, base64, io, json, os, sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# -----------------------------
# Utilities
# -----------------------------

def b64png(pil_img) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def clip_bbox(b: Tuple[float,float,float,float], w: float, h: float):
    x0,y0,x1,y1 = b
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return [float(x0), float(y0), float(x1), float(y1)]

def ensure_module(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False

# -----------------------------
# Data models
# -----------------------------

@dataclass
class Word:
    id: int
    text: str
    bbox: List[float]   # [x0,y0,x1,y1]
    conf: float = 1.0

@dataclass
class Line:
    id: int
    text: str
    bbox: List[float]
    conf: float = 1.0

@dataclass
class PageOut:
    page_index: int
    width: float
    height: float
    image_bytes: Optional[str]  # base64 PNG or None
    lines: List[Line]
    words: List[Word]

@dataclass
class DocOut:
    doc_id: str
    backend: str
    pages: List[PageOut]

# -----------------------------
# Backend: PyMuPDF digital text
# -----------------------------
def run_pymupdf_text(pdf_path: str, max_pages: Optional[int], embed_images: bool, dpi: int) -> DocOut:
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open(pdf_path)
    out_pages: List[PageOut] = []
    N = len(doc)
    n_pages = min(N, max_pages) if max_pages else N

    for i in range(n_pages):
        page = doc[i]
        w, h = float(page.rect.width), float(page.rect.height)

        # words: list of (x0,y0,x1,y1,"text", block_no, line_no, word_no)
        words_raw = page.get_text("words")
        words: List[Word] = []
        for wi, wr in enumerate(words_raw):
            x0, y0, x1, y1, txt, *_ = wr
            words.append(Word(id=wi, text=txt or "", bbox=clip_bbox((x0,y0,x1,y1), w, h), conf=1.0))

        # merge into lines by line index as PyMuPDF reports
        lines: List[Line] = []
        # Using detailed dict gives structured blocks/lines
        textdict = page.get_text("dict")
        li = 0
        for b in textdict.get("blocks", []):
            for l in b.get("lines", []):
                span_txts = []
                xs, ys, xe, ye = 1e9,1e9,-1,-1
                for s in l.get("spans", []):
                    span_txts.append(s.get("text",""))
                    x0,y0,x1,y1 = s.get("bbox", [0,0,0,0])
                    xs, ys = min(xs,x0), min(ys,y0)
                    xe, ye = max(xe,x1), max(ye,y1)
                txt = " ".join([t for t in span_txts if t])
                if txt.strip():
                    lines.append(Line(id=li, text=txt, bbox=clip_bbox((xs,ys,xe,ye), w, h), conf=1.0))
                    li += 1

        img_b64 = None
        if embed_images:
            # render page to PNG at given DPI
            zoom = dpi/72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_b64 = b64png(pil)

        out_pages.append(PageOut(
            page_index=i, width=w, height=h,
            image_bytes=img_b64, lines=lines, words=words
        ))

    return DocOut(doc_id=os.path.basename(pdf_path), backend="pymupdf_text", pages=out_pages)

# -----------------------------
# Backend: pdfminer (lines only)
# -----------------------------
def run_pdfminer(pdf_path: str, max_pages: Optional[int], embed_images: bool, dpi: int) -> DocOut:
    # pdfminer.six: basic text; bbox by LTTextLine
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTPage, LTTextContainer, LTTextLine
    from PIL import Image
    import fitz

    out_pages: List[PageOut] = []
    # we still use PyMuPDF (if available) for optional page raster
    have_fitz = ensure_module("fitz")
    fitz_doc = fitz.open(pdf_path) if have_fitz else None

    for pi, page_layout in enumerate(extract_pages(pdf_path)):
        if max_pages and pi >= max_pages: break
        assert isinstance(page_layout, LTPage)
        w, h = float(page_layout.width), float(page_layout.height)
        lines: List[Line] = []
        li = 0
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for obj in element:
                    if isinstance(obj, LTTextLine):
                        x0,y0,x1,y1 = obj.bbox
                        txt = obj.get_text().strip()
                        if txt:
                            lines.append(Line(id=li, text=txt, bbox=clip_bbox((x0,y0,x1,y1), w, h)))
                            li += 1
        img_b64 = None
        if embed_images and have_fitz:
            from PIL import Image
            page = fitz_doc[pi]
            zoom = dpi/72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_b64 = b64png(pil)

        out_pages.append(PageOut(page_index=pi, width=w, height=h, image_bytes=img_b64, lines=lines, words=[]))

    return DocOut(doc_id=os.path.basename(pdf_path), backend="pdfminer", pages=out_pages)

# -----------------------------
# Backend: EasyOCR (OCR on images)
# -----------------------------
def run_easyocr(pdf_path: str, max_pages: Optional[int], embed_images: bool, dpi: int) -> DocOut:
    import fitz
    from PIL import Image
    import numpy as np
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    doc = fitz.open(pdf_path)
    out_pages: List[PageOut] = []
    N = len(doc)
    n_pages = min(N, max_pages) if max_pages else N

    for i in range(n_pages):
        page = doc[i]
        # raster to image for OCR
        zoom = dpi/72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr = np.array(pil)

        # EasyOCR returns list of [bbox, text, conf]
        result = reader.readtext(arr)
        words: List[Word] = []
        lines: List[Line] = []
        wi = 0
        for (bx, by, bx2, by2), txt, conf in [(_flatten_quad(b), t, c) for b,t,c in result]:
            bbox = clip_bbox((bx, by, bx2, by2), arr.shape[1], arr.shape[0])
            words.append(Word(id=wi, text=txt, bbox=bbox, conf=float(conf)))
            wi += 1
        # simple line grouping: join nearby words on same y-band
        lines = _group_words_to_lines(words)
        img_b64 = b64png(pil) if embed_images else None
        out_pages.append(PageOut(page_index=i, width=arr.shape[1], height=arr.shape[0],
                                 image_bytes=img_b64, lines=lines, words=words))
    return DocOut(doc_id=os.path.basename(pdf_path), backend="easyocr", pages=out_pages)

def _flatten_quad(poly):
    # EasyOCR gives 4 points; return min/max rectangle
    xs = [poly[0][0], poly[1][0], poly[2][0], poly[3][0]]
    ys = [poly[0][1], poly[1][1], poly[2][1], poly[3][1]]
    return min(xs), min(ys), max(xs), max(ys)

def _group_words_to_lines(words: List[Word], y_merge_tol: float = 8.0) -> List[Line]:
    if not words: return []
    # naive: cluster by y-center bands
    bands: List[Tuple[float, List[Word]]] = []
    for w in words:
        _, y0, _, y1 = w.bbox; yc = 0.5*(y0+y1)
        placed = False
        for (y, arr) in bands:
            if abs(yc - y) <= y_merge_tol:
                arr.append(w); placed = True; break
        if not placed:
            bands.append((yc, [w]))
    lines: List[Line] = []
    li = 0
    for _, arr in sorted(bands, key=lambda t: t[0]):
        arr = sorted(arr, key=lambda w: w.bbox[0])
        txt = " ".join([w.text for w in arr if w.text])
        xs = min(w.bbox[0] for w in arr); ys = min(w.bbox[1] for w in arr)
        xe = max(w.bbox[2] for w in arr); ye = max(w.bbox[3] for w in arr)
        lines.append(Line(id=li, text=txt, bbox=[xs,ys,xe,ye], conf=sum(w.conf for w in arr)/max(1,len(arr))))
        li += 1
    return lines

# -----------------------------
# Backend: PaddleOCR (OCR on images)
# -----------------------------
def run_paddleocr(pdf_path: str, max_pages: Optional[int], embed_images: bool, dpi: int) -> DocOut:
    import fitz
    from PIL import Image
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False, use_gpu=False)
    doc = fitz.open(pdf_path)
    out_pages: List[PageOut] = []
    N = len(doc)
    n_pages = min(N, max_pages) if max_pages else N

    for i in range(n_pages):
        page = doc[i]
        zoom = dpi/72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr = pil  # PaddleOCR accepts file path or numpy; we’ll use PIL via temp bytes
        # Use bytes buffer
        buf = io.BytesIO(); pil.save(buf, format="PNG")
        buf.seek(0)
        result = ocr.ocr(buf.getvalue(), cls=True)

        words: List[Word] = []
        wi = 0
        if result and result[0]:
            for line in result[0]:
                # line: [ [ [x,y],...4 ], (text, conf) ]
                box, (txt, conf) = line
                xs = [p[0] for p in box]; ys = [p[1] for p in box]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                words.append(Word(id=wi, text=txt, bbox=bbox, conf=float(conf)))
                wi += 1

        lines = _group_words_to_lines(words)
        img_b64 = b64png(pil) if embed_images else None
        out_pages.append(PageOut(page_index=i, width=pil.width, height=pil.height,
                                 image_bytes=img_b64, lines=lines, words=words))
    return DocOut(doc_id=os.path.basename(pdf_path), backend="paddleocr", pages=out_pages)

# -----------------------------
# Main
# -----------------------------
def choose_backend(name: Optional[str]) -> str:
    # order of preference
    if name: return name
    # default: pymupdf_text
    return "pymupdf_text"

def main():
    p = argparse.ArgumentParser(description="LexiGraph OCR/Text extractor")
    p.add_argument("pdf", help="Path to PDF")
    p.add_argument("-o", "--out", default=None, help="Path to write JSON (default: stdout)")
    p.add_argument("--backend", default=os.getenv("LEXI_OCR_BACKEND", None),
                   choices=["pymupdf_text","pdfminer","easyocr","paddleocr"],
                   help="Which backend to use (default: pymupdf_text)")
    p.add_argument("--max-pages", type=int, default=None, help="Limit number of pages")
    p.add_argument("--dpi", type=int, default=200, help="Raster DPI when OCR/rasterizing")
    p.add_argument("--embed-images", action="store_true", help="Embed page PNGs in JSON")
    args = p.parse_args()

    backend = choose_backend(args.backend)

    # graceful missing-module messages
    if backend == "pymupdf_text" and not ensure_module("fitz"):
        print("ERROR: PyMuPDF (fitz) not installed. Try: pip install pymupdf", file=sys.stderr)
        sys.exit(2)
    if backend == "pdfminer" and not ensure_module("pdfminer"):
        print("ERROR: pdfminer.six not installed. Try: pip install pdfminer.six", file=sys.stderr)
        sys.exit(2)
    if backend == "easyocr" and not ensure_module("easyocr"):
        print("ERROR: easyocr not installed. Try: pip install easyocr", file=sys.stderr)
        sys.exit(2)
    if backend == "paddleocr" and not ensure_module("paddleocr"):
        print("ERROR: paddleocr not installed. Try: pip install paddleocr", file=sys.stderr)
        sys.exit(2)

    if backend == "pymupdf_text":
        doc = run_pymupdf_text(args.pdf, args.max_pages, args.embed_images, args.dpi)
    elif backend == "pdfminer":
        doc = run_pdfminer(args.pdf, args.max_pages, args.embed_images, args.dpi)
    elif backend == "easyocr":
        doc = run_easyocr(args.pdf, args.max_pages, args.embed_images, args.dpi)
    elif backend == "paddleocr":
        doc = run_paddleocr(args.pdf, args.max_pages, args.embed_images, args.dpi)
    else:
        print(f"ERROR: Unknown backend {backend}", file=sys.stderr)
        sys.exit(2)

    payload = {"doc_id": doc.doc_id, "backend": doc.backend, "pages": [asdict(p) for p in doc.pages]}
    js = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(js)
        print(f"✔ Wrote {args.out}")
    else:
        print(js)

if __name__ == "__main__":
    main()
