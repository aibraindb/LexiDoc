# main.py
import io, os, base64, json, math, tempfile, time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import numpy as np
from PIL import Image
import fitz  # PyMuPDF

def _has(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False

HAVE_PDFMINER = _has("pdfminer")
HAVE_PYTESS   = _has("pytesseract")
HAVE_PADDLE   = _has("paddleocr")

# ---------------- Data models ----------------
@dataclass
class Word:
    id: int
    text: str
    bbox: List[float]
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
    image_bytes: Optional[str]  # base64 PNG (preview)
    lines: List[Line]
    words: List[Word]

@dataclass
class DocOut:
    doc_id: str
    backend: str
    pages: List[PageOut]

def _clip(b, w, h):
    x0,y0,x1,y1 = b
    x0 = max(0, min(x0, w)); x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h)); y1 = max(0, min(y1, h))
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    return [float(x0), float(y0), float(x1), float(y1)]

def _raster_page(p: fitz.Page, dpi: int) -> Image.Image:
    zoom = (dpi/72.0)
    mat = fitz.Matrix(zoom, zoom)
    pix = p.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

def _pil_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------- Backends ----------------
def run_pymupdf_text(pdf_path: str, max_pages: Optional[int]=None, embed_images: bool=True, dpi: int=200) -> DocOut:
    doc = fitz.open(pdf_path)
    pages: List[PageOut] = []
    n = min(len(doc), max_pages or len(doc))
    for i in range(n):
        p = doc[i]
        w,h = float(p.rect.width), float(p.rect.height)
        # words
        words_raw = p.get_text("words")
        words = [Word(id=j, text=(wr[4] or ""), bbox=_clip(wr[:4], w, h), conf=1.0)
                 for j, wr in enumerate(words_raw)]
        # lines
        lines: List[Line] = []
        li=0
        td = p.get_text("dict")
        for b in td.get("blocks", []):
            for l in b.get("lines", []):
                xs, ys, xe, ye = 1e9, 1e9, -1, -1
                parts=[]
                for s in l.get("spans", []):
                    parts.append(s.get("text",""))
                    x0,y0,x1,y1 = s.get("bbox", [0,0,0,0])
                    xs, ys = min(xs,x0), min(ys,y0)
                    xe, ye = max(xe,x1), max(ye,y1)
                txt = " ".join([t for t in parts if t]).strip()
                if txt:
                    lines.append(Line(id=li, text=txt, bbox=_clip((xs,ys,xe,ye), w, h)))
                    li += 1
        img_b64 = _pil_png_b64(_raster_page(p, dpi)) if embed_images else None
        pages.append(PageOut(page_index=i, width=w, height=h, image_bytes=img_b64, lines=lines, words=words))
    return DocOut(doc_id=os.path.basename(pdf_path), backend="pymupdf_text", pages=pages)

def run_pdfminer(pdf_path: str, max_pages: Optional[int]=None, embed_images: bool=True, dpi: int=200) -> DocOut:
    if not HAVE_PDFMINER:
        raise RuntimeError("pdfminer.six not installed")
    from pdfminer.high_level import extract_text
    doc = fitz.open(pdf_path)
    pages=[]
    text_all = extract_text(pdf_path) or ""
    parts = [t.strip() for t in text_all.splitlines() if t.strip()]
    for i in range(min(len(doc), max_pages or len(doc))):
        p = doc[i]; w,h = float(p.rect.width), float(p.rect.height)
        lines = [Line(id=j, text=parts[j], bbox=[0, 20*j, w, 20*(j+1)]) for j in range(min(100, len(parts)))]
        img_b64 = _pil_png_b64(_raster_page(p, dpi)) if embed_images else None
        pages.append(PageOut(page_index=i, width=w, height=h, image_bytes=img_b64, lines=lines, words=[]))
    return DocOut(doc_id=os.path.basename(pdf_path), backend="pdfminer", pages=pages)

def run_tesseract(pdf_path: str, max_pages: Optional[int]=None, embed_images: bool=True, dpi: int=200) -> DocOut:
    if not HAVE_PYTESS:
        raise RuntimeError("pytesseract not installed")
    import pytesseract
    tess_exe = os.environ.get("TESSERACT_EXE")
    if tess_exe: pytesseract.pytesseract.tesseract_cmd = tess_exe

    doc = fitz.open(pdf_path)
    pages=[]
    n = min(len(doc), max_pages or len(doc))
    for i in range(n):
        p = doc[i]; w,h = float(p.rect.width), float(p.rect.height)
        img = _raster_page(p, dpi)
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words=[]
        for j in range(len(data["text"])):
            txt = (data["text"][j] or "").strip()
            if not txt: continue
            x, y, bw, bh = data["left"][j], data["top"][j], data["width"][j], data["height"][j]
            conf_j = data.get("conf",[0]*len(data["text"]))[j]
            try: conf_f = float(conf_j)
            except: conf_f = 0.0
            words.append(Word(
                id=j, text=txt,
                bbox=_clip([x/dpi*72, y/dpi*72, (x+bw)/dpi*72, (y+bh)/dpi*72], w, h),
                conf=conf_f
            ))
        # group to lines
        line_map={}
        for j in range(len(data["text"])):
            key = (data.get("block_num",[0])[j], data.get("line_num",[0])[j])
            line_map.setdefault(key, []).append(j)
        lines=[]
        li=0
        for key, idxs in line_map.items():
            xs,ys,xe,ye = 1e9,1e9,-1,-1
            parts=[]
            for j in idxs:
                txt=(data["text"][j] or "").strip()
                if not txt: continue
                x,y,bw,bh = data["left"][j], data["top"][j], data["width"][j], data["height"][j]
                xs,ys = min(xs,x), min(ys,y); xe,ye=max(xe,x+bw), max(ye,y+bh)
                parts.append(txt)
            if parts:
                bbox72 = [xs/dpi*72, ys/dpi*72, xe/dpi*72, ye/dpi*72]
                lines.append(Line(id=li, text=" ".join(parts), bbox=_clip(bbox72, w, h)))
                li+=1
        img_b64 = _pil_png_b64(img) if embed_images else None
        pages.append(PageOut(page_index=i, width=w, height=h, image_bytes=img_b64, lines=lines, words=words))
    return DocOut(doc_id=os.path.basename(pdf_path), backend="tesseract", pages=pages)

def run_paddleocr(pdf_path: str, max_pages: Optional[int]=None, embed_images: bool=True, dpi: int=200) -> DocOut:
    if not HAVE_PADDLE:
        raise RuntimeError("paddleocr not installed")
    from paddleocr import PaddleOCR
    # Minimal, CPU-safe init across versions
    ocr = PaddleOCR(lang="en", show_log=False)
    doc = fitz.open(pdf_path)
    pages=[]
    n = min(len(doc), max_pages or len(doc))
    for i in range(n):
        p = doc[i]; w,h = float(p.rect.width), float(p.rect.height)
        img = _raster_page(p, dpi)
        res = ocr.ocr(np.array(img))
        lines=[]; li=0
        # res could be list[list[(poly, (text, conf))]] or list[(poly,(text,conf))]
        blocks = []
        if isinstance(res, list) and len(res)>0 and isinstance(res[0], list):
            for blk in res:
                blocks.extend(blk or [])
        elif isinstance(res, list):
            blocks = res
        for entry in (blocks or []):
            try:
                box, data = entry
                txt, conf = data
                xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
                x0,y0,x1,y1 = min(xs),min(ys),max(xs),max(ys)
                bbox72 = [x0/dpi*72, y0/dpi*72, x1/dpi*72, y1/dpi*72]
                lines.append(Line(id=li, text=(txt or ""), bbox=_clip(bbox72, w, h), conf=float(conf or 0)))
                li+=1
            except Exception:
                continue
        img_b64 = _pil_png_b64(img) if embed_images else None
        pages.append(PageOut(page_index=i, width=w, height=h, image_bytes=img_b64, lines=lines, words=[]))
    return DocOut(doc_id=os.path.basename(pdf_path), backend="paddleocr", pages=pages)

# ---------------- Mini vector index ----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = "data/doc_index"; os.makedirs(INDEX_DIR, exist_ok=True)
IDX_JSON = os.path.join(INDEX_DIR, "meta.json")
VEC_NPY  = os.path.join(INDEX_DIR, "vecs.npy")

def _load_index():
    if not os.path.exists(IDX_JSON): return {"docs": []}, None
    with open(IDX_JSON, "r") as f:
        meta = json.load(f)
    X = np.load(VEC_NPY) if os.path.exists(VEC_NPY) else None
    return meta, X

def _save_index(meta, X):
    with open(IDX_JSON, "w") as f: json.dump(meta, f)
    if X is not None: np.save(VEC_NPY, X)

def _doc_text(doc: DocOut) -> str:
    parts=[]
    for p in doc.pages:
        for ln in p.lines:
            if ln.text: parts.append(ln.text)
    return "\n".join(parts)

# --- REPLACE upsert_document IN src/tbd/main.py ---

def upsert_document(doc: DocOut):
    meta, _ = _load_index()

    # 1) Normalize / build corpus
    def _norm(t: str) -> str:
        t = (t or "").strip()
        return t if t else "[[EMPTY-DOC]]"

    corpus = [_norm(d.get("text", "")) for d in meta.get("docs", [])]
    new_text = _norm(_doc_text(doc))
    corpus.append(new_text)

    # Optional: drop exact duplicates to keep the vectorizer stable
    # (but still store the new doc in meta)
    corpus_unique = list(dict.fromkeys(corpus))  # preserves order

    n_docs = len(corpus_unique)

    # 2) Choose safe TF-IDF params based on corpus size
    if n_docs <= 2:
        # 1–2 docs: never prune by df
        vec = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 2))
    elif n_docs <= 5:
        # very small: prune almost nothing
        vec = TfidfVectorizer(min_df=1, max_df=0.9999, ngram_range=(1, 2))
    else:
        vec = TfidfVectorizer(min_df=1, max_df=0.95, ngram_range=(1, 2))

    # 3) Fit with safety fallbacks
    try:
        X = vec.fit_transform(corpus_unique)
        if X.shape[1] == 0:
            raise ValueError("no terms after pruning")
        X = X.toarray().astype(np.float32)
    except Exception:
        # Back off to the most permissive settings
        vec = TfidfVectorizer(min_df=1, max_df=1.0, ngram_range=(1, 1))
        X = vec.fit_transform(corpus_unique).toarray().astype(np.float32)

    # 4) Map unique-matrix rows back to full corpus length if you need 1:1,
    #    otherwise just store vectors for the unique set. Here we re-compute
    #    for the full corpus using the fitted vocabulary so shapes align.
    X_full = TfidfVectorizer(vocabulary=vec.vocabulary_).fit_transform(corpus).toarray().astype(np.float32)

    # 5) Persist meta + vectors
    meta.setdefault("docs", [])
    meta["docs"].append({
        "doc_id": doc.doc_id,
        "backend": doc.backend,
        "text": new_text,
        "ts": time.time(),
    })

    _save_index(meta, X_full)


def stats():
    meta, X = _load_index()
    return {"num_docs": len(meta["docs"]), "vec_shape": None if X is None else list(X.shape)}

# ---------------- FIBO (minimal) ----------------
FIBO_INDEX = "data/fibo_index.json"
def fibo_search(q: str, limit=25):
    if not os.path.exists(FIBO_INDEX): return []
    idx = json.loads(open(FIBO_INDEX).read())
    classes = idx.get("classes", [])
    labels = [(c.get("label") or c["uri"].split("/")[-1]) for c in classes]
    ns     = [c.get("ns","") for c in classes]
    vec = TfidfVectorizer(min_df=1).fit(labels + [q])
    qv = vec.transform([q]).toarray()
    Cv = vec.transform(labels).toarray()
    sims = cosine_similarity(qv, Cv)[0]
    order = np.argsort(-sims)[:limit]
    out=[]
    for i in order:
        out.append({"uri": classes[i]["uri"], "label": labels[i], "ns": ns[i], "score": float(sims[i])})
    return out

# ---------------- FastAPI ----------------
app = FastAPI()

class ExtractResp(BaseModel):
    doc: dict

@app.post("/extract", response_model=ExtractResp)
async def extract(
    file: UploadFile = File(...),
    backend: str = Form("pymupdf_text"),
    max_pages: int = Form(3),
    embed_images: bool = Form(True),
    dpi: int = Form(200),
    index_after: bool = Form(False),
):
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data); tmp_path = tmp.name
    if backend == "pymupdf_text":
        doc = run_pymupdf_text(tmp_path, max_pages, embed_images, dpi)
    elif backend == "pdfminer":
        doc = run_pdfminer(tmp_path, max_pages, embed_images, dpi)
    elif backend == "tesseract":
        doc = run_tesseract(tmp_path, max_pages, embed_images, dpi)
    elif backend == "paddleocr":
        doc = run_paddleocr(tmp_path, max_pages, embed_images, dpi)
    else:
        return JSONResponse({"error":"unknown backend"}, status_code=400)
    payload = {"doc_id": doc.doc_id, "backend": doc.backend, "pages": [asdict(p) for p in doc.pages]}
    if index_after:
        upsert_document(doc)
    return {"doc": payload}

@app.post("/index/upsert")
async def index_upsert(doc: dict):
    pages = []
    for p in doc.get("pages", []):
        lines = [Line(**ln) for ln in p.get("lines",[])]
        words = [Word(**w) for w in p.get("words",[])]
        pages.append(PageOut(page_index=p["page_index"], width=p["width"], height=p["height"], image_bytes=p.get("image_bytes"), lines=lines, words=words))
    d = DocOut(doc_id=doc["doc_id"], backend=doc["backend"], pages=pages)
    upsert_document(d)
    return {"status":"ok"}

@app.get("/index/stats")
async def index_stats():
    return stats()

@app.get("/fibo/search")
async def api_fibo_search(q: str, limit: int = 25):
    return fibo_search(q, limit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
