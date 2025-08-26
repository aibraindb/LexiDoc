# ui_app.py
import io, os, json, base64, time
from dataclasses import asdict
from typing import List, Dict
import streamlit as st
from PIL import Image, ImageDraw

# local imports
from main import (
    run_pymupdf_text, run_pdfminer, run_tesseract, run_paddleocr,
    fibo_search, stats, upsert_document
)

st.set_page_config(page_title="LexiGraph – Doc Viewer", layout="wide")

st.title("LexiGraph – Document Viewer + FIBO lookup")

# -------- Sidebar controls --------
with st.sidebar:
    st.header("Extraction")
    backend = st.selectbox("Backend", ["pymupdf_text","pdfminer","tesseract","paddleocr"], index=0)
    max_pages = st.number_input("Max pages", 1, 200, 3)
    dpi = st.slider("DPI (preview & OCR)", 100, 360, 200, step=10)
    embed_images = st.checkbox("Embed page previews", True)

    st.header("Display")
    layer = st.radio("Annotate", ["Lines","Words"], horizontal=True)
    show_json = st.checkbox("Show selected JSON", True)

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

colL, colR = st.columns([1.1, 1.9], gap="medium")

# -------- Left: FIBO & doc table --------
with colL:
    st.subheader("FIBO search")
    q = st.text_input("Search FIBO (free text):", placeholder="e.g., lease, invoice, guarantee…")
    if q:
        hits = fibo_search(q, limit=15)
        for h in hits:
            st.markdown(f"- **{h['label']}**  \n  `{h['ns']}`")

    st.subheader("Indexed documents")
    idx = stats()
    st.caption(f"Indexed docs: **{idx['num_docs']}** — vec shape: {idx['vec_shape']}")
    links_path = "data/doc_links.json"; os.makedirs("data", exist_ok=True)
    links: Dict[str,str] = {}
    if os.path.exists(links_path):
        try:
            links = json.loads(open(links_path).read())
        except Exception:
            links = {}

# -------- Right: Viewer --------
with colR:
    if not uploaded:
        st.info("Upload a PDF to extract and view bounding boxes.")
        st.stop()

    if st.button("Extract"):
        path = "/tmp/_lexi_ui.pdf"
        with open(path, "wb") as f: f.write(uploaded.read())
        if backend=="pymupdf_text":
            doc = run_pymupdf_text(path, max_pages, embed_images, dpi)
        elif backend=="pdfminer":
            doc = run_pdfminer(path, max_pages, embed_images, dpi)
        elif backend=="tesseract":
            doc = run_tesseract(path, max_pages, embed_images, dpi)
        else:
            doc = run_paddleocr(path, max_pages, embed_images, dpi)
        st.session_state["doc_json"] = {"doc_id": doc.doc_id, "backend": doc.backend, "pages": [asdict(p) for p in doc.pages]}
        # upsert to local index
        upsert_document(doc)
        st.success(f"Extracted {doc.doc_id} with {doc.backend} and indexed.")

    dj = st.session_state.get("doc_json")
    if not dj:
        st.stop()

    pages = dj["pages"]
    page_idx = st.number_input("Page", 1, len(pages), 1) - 1
    pg = pages[page_idx]
    st.caption(f"Page {page_idx+1} • {int(pg['width'])}×{int(pg['height'])}pt • {len(pg['lines'])} lines, {len(pg['words'])} words")

    # page image
    if not pg.get("image_bytes"):
        st.warning("No preview. Re-extract with 'Embed page previews' enabled.")
        st.stop()
    img = Image.open(io.BytesIO(base64.b64decode(pg["image_bytes"])))

    # items
    items = pg["lines"] if layer=="Lines" else pg["words"]
    # dropdown (labels precomputed to avoid f-string backslash issue)
    labels = []
    for it in items:
        txt = (it.get("text","") or "").replace("\n", " ")
        if len(txt) > 60: txt = txt[:60] + "…"
        labels.append(f"[{it['id']}] {txt}")
    default_idx = 0 if items else 0
    sel_label = st.selectbox("Select item", options=labels if labels else ["—"], index=default_idx if labels else 0)
    selected = None
    if items and labels:
        sel_idx = labels.index(sel_label)
        selected = items[sel_idx]

    # draw boxes (selected green; others red)
    draw = ImageDraw.Draw(img)
    W, H = img.width, img.height
    sx = W / max(1e-6, pg["width"]); sy = H / max(1e-6, pg["height"])

    def to_px(b):
        x0,y0,x1,y1 = b
        return [x0*sx, y0*sy, x1*sx, y1*sy]

    for it in items:
        x0,y0,x1,y1 = to_px(it["bbox"])
        draw.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=1)

    if selected:
        x0,y0,x1,y1 = to_px(selected["bbox"])
        draw.rectangle([x0,y0,x1,y1], outline=(0,200,0), width=3)

    st.image(img, use_column_width=True)

    if show_json and selected:
        st.json({
            "doc_id": dj["doc_id"],
            "page_index": pg["page_index"],
            "item": {
                "id": selected["id"],
                "text": selected.get("text",""),
                "bbox_points": selected["bbox"],  # pt units
            }
        })

    # simple editable FIBO link per document
    st.subheader("Link this document to a FIBO class")
    current_link = links.get(dj["doc_id"], "")
    new_link = st.text_input("FIBO class URI", value=current_link, placeholder="Paste a FIBO class URI here…")
    if st.button("Save Document ↔ FIBO Link"):
        links[dj["doc_id"]] = new_link
        with open(links_path, "w") as f: json.dump(links, f, indent=2)
        st.success("Saved.")

    st.download_button("Download full JSON", json.dumps(dj, indent=2), file_name=f"{dj['doc_id']}.json", mime="application/json")
