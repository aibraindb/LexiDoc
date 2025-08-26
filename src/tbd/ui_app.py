# ui_app.py
# Streamlit UI for: document indexing + FIBO search / subgraph
# -----------------------------------------------------------
import json
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# --- IMPORTANT: call set_page_config FIRST ---
st.set_page_config(page_title="LexiGraph — Docs & FIBO", layout="wide")

# ---------- pluggable imports (graceful fallbacks) ----------
# core indexing & FIBO search live here:
try:
    from tbd.main import upsert_document, stats, fibo_search
except Exception as e:  # pragma: no cover
    upsert_document = None
    stats = None
    fibo_search = None

# text extraction backends (optional; we guard each):
try:
    from tbd.main import run_pdfminer, run_easyocr, run_paddleocr
except Exception:
    run_pdfminer = run_easyocr = run_paddleocr = None

# FIBO subgraph builder:
try:
    from tbd.core.fibo_graph import build_subgraph
except Exception:
    build_subgraph = None

# RDF namespaces (used inside build_subgraph; having them here avoids NameError in some envs)
try:
    from rdflib import RDF, RDFS, OWL  # noqa: F401
except Exception:
    pass

# ---------- helpers ----------
def _bytes_to_dataurl(img_bytes: bytes, mime="image/png") -> str:
    b64 = base64.b64encode(img_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

def _d3_html(graph_json: dict, fit: bool, reset: bool, tag: str) -> str:
    """Inline D3 graph; tag guarantees a unique DOM id per draw."""
    data = json.dumps(graph_json or {"nodes": [], "links": []})
    div_id = f"d3g_{tag}"

    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  html,body,#{div_id}{{margin:0;padding:0;width:100%;height:100%;}}
  .node circle{{fill:#eef3ff;stroke:#4666ff;stroke-width:1.4px}}
  .node text{{font:12px system-ui;pointer-events:none}}
  .link{{stroke:#b9c2d0;stroke-opacity:.7}}
</style>
</head>
<body>
<div id="{div_id}"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const holder = document.getElementById("{div_id}");
const data   = {data};

function render(){{
  holder.innerHTML = "";
  const w = holder.clientWidth || 800;
  const h = holder.clientHeight || 600;

  const svg = d3.select(holder).append("svg")
      .attr("width", w)
      .attr("height", h);

  const g = svg.append("g");

  const zoom = d3.zoom().scaleExtent([0.2, 4]).on("zoom", (e)=>g.attr("transform", e.transform));
  svg.call(zoom);

  const link = g.selectAll(".link")
    .data(data.links || [])
    .enter().append("line")
    .attr("class","link")
    .attr("stroke-width", d => d.weight ? Math.max(1, Math.min(4, d.weight)) : 1);

  const node = g.selectAll(".node")
    .data(data.nodes || [])
    .enter().append("g")
      .attr("class","node")
      .call(d3.drag()
          .on("start", (e,d)=>{{ if(!e.active) sim.alphaTarget(.3).restart(); d.fx=d.x; d.fy=d.y; }})
          .on("drag",  (e,d)=>{{ d.fx=e.x; d.fy=e.y; }})
          .on("end",   (e,d)=>{{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}));

  node.append("circle").attr("r", 12);
  node.append("text").attr("x", 16).attr("y", 4).text(d => d.label || d.id);

  const sim = d3.forceSimulation(data.nodes || [])
    .force("link", d3.forceLink(data.links || []).id(d => d.id).distance(120).strength(0.85))
    .force("charge", d3.forceManyBody().strength(-280))
    .force("center", d3.forceCenter(w/2, h/2))
    .on("tick", ()=>{{
      link
        .attr("x1", d=>d.source.x).attr("y1", d=>d.source.y)
        .attr("x2", d=>d.target.x).attr("y2", d=>d.target.y);
      node
        .attr("transform", d=>`translate(${{d.x}},${{d.y}})`);
    }});

  function fitToView(){{
    if (!(data.nodes && data.nodes.length)) return;
    let minX=Infinity, minY=Infinity, maxX=-Infinity, maxY=-Infinity;
    (data.nodes||[]).forEach(n=>{{minX=Math.min(minX,n.x); minY=Math.min(minY,n.y); maxX=Math.max(maxX,n.x); maxY=Math.max(maxY,n.y);}});
    const padding=40, dx=(maxX-minX)+padding*2, dy=(maxY-minY)+padding*2;
    const sx=w/dx, sy=h/dy, s=Math.max(0.2, Math.min(4, Math.min(sx, sy)));
    const tx=(w - s*(minX+maxX))/2, ty=(h - s*(minY+maxY))/2;
    svg.transition().duration(450).call(zoom.transform, d3.zoomIdentity.translate(tx,ty).scale(s));
  }}

  if ({str(bool(fit)).lower()}) fitToView();
  if ({str(bool(reset)).lower()}) svg.transition().duration(250).call(zoom.transform, d3.zoomIdentity);

  const ro = new ResizeObserver(()=>render());
  ro.observe(holder);
}}
render();
</script>
</body>
</html>
"""

# ---------- header ----------
st.title("LexiGraph — Documents & FIBO")

# ---------- tabs ----------
tab_docs, tab_fibo = st.tabs(["📄 Documents", "🔎 FIBO"])

# =========================
# TAB: DOCUMENTS
# =========================
with tab_docs:
    st.subheader("Upload & index")

    colU, colOpts = st.columns([2,1])
    with colU:
        up = st.file_uploader("PDF to index", type=["pdf"], key="pdf_upl")
    with colOpts:
        backend = st.selectbox(
            "Extraction backend",
            ["pdfminer", "easyocr", "paddleocr"],
            index=0,
            help="Use PDFMiner for digital PDFs; OCR backends for scans.",
            key="backend_sel",
        )
        dpi   = st.slider("Render DPI (OCR)", 96, 300, 144, key="dpi_sel")
        maxp  = st.number_input("Max pages", 1, 200, 12, step=1, key="maxpages_sel")
        embed = st.checkbox("Embed first-page image preview", value=True, key="img_embed_sel")

    if up and upsert_document:
        path = None
        # Some backends want bytes; others are fine with a saved tmp file.
        pdf_bytes = up.read()
        # Route to chosen backend (gracefully warn if missing)
        if backend == "pdfminer":
            if run_pdfminer:
                doc = run_pdfminer(pdf_bytes, int(maxp), bool(embed), int(dpi))
            else:
                st.error("PDFMiner backend not available in this environment.")
                doc = None
        elif backend == "easyocr":
            if run_easyocr:
                doc = run_easyocr(pdf_bytes, int(maxp), bool(embed), int(dpi))
            else:
                st.error("EasyOCR backend not available.")
                doc = None
        else:  # paddleocr
            if run_paddleocr:
                doc = run_paddleocr(pdf_bytes, int(maxp), bool(embed), int(dpi))
            else:
                st.error("PaddleOCR backend not available.")
                doc = None

        if doc:
            # upsert + echo summary
            try:
                upsert_document(doc)
                st.success(f"Indexed: **{doc.get('name','(unnamed)')}**  ·  chars: {len(doc.get('text','')):,}")
            except Exception as e:
                st.error(f"Indexing failed: {e}")

            # preview
            p0 = (doc.get("pages") or [{}])[0]
            img_b = p0.get("image_bytes")
            if img_b and embed:
                st.image(BytesIO(img_b), caption="page 1 preview", use_column_width=True)

    st.subheader("Indexed documents")
    if stats:
        try:
            s = stats()
        except Exception as e:
            s = {}
            st.warning(f"stats() failed: {e}")
    else:
        s = {}

    docs = s.get("documents", [])
    if docs:
        st.dataframe(
            [
                {
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "chars": len(d.get("text","")),
                    "pages": len(d.get("pages") or []),
                    "when": d.get("ts"),
                }
                for d in docs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No documents in the index yet.")

# =========================
# TAB: FIBO
# =========================
with tab_fibo:
    st.subheader("FIBO explorer")

    # keep the TTL path in session for build_subgraph
    fibo_ttl = st.text_input(
        "FIBO TTL path",
        value=st.session_state.get("fibo_ttl_path", "fibo_full.ttl"),
        help="Absolute or relative path to fibo_full.ttl",
        key="ttl_path_box",
    )
    st.session_state["fibo_ttl_path"] = fibo_ttl

    q = st.text_input("Search (labels / altLabels / camelCase)", key="fibo_q_box")
    hops = st.slider("Neighborhood hops", 1, 4, 2, key="fibo_hops_box")

    if not fibo_search:
        st.error("fibo_search() not available. Make sure tbd.main exposes it.")
    else:
        if q:
            hits = fibo_search(q.strip(), limit=15)
            if not hits:
                st.info("No matches.")
            else:
                colL, colR = st.columns([1, 2], gap="large")

                with colL:
                    st.caption("Results")
                    options = [h["uri"] for h in hits]
                    labels  = {
                        h["uri"]: f'{h.get("label") or "(no label)"} · {h.get("score",0):.3f}' for h in hits
                    }
                    chosen_uri = st.radio(
                        "Pick one to visualize",
                        options,
                        format_func=lambda u: labels[u],
                        key="fibo_pick_radio",
                    )
                    # tiny metadata
                    if chosen_uri:
                        st.write(chosen_uri)

                with colR:
                    st.caption("Subgraph")
                    if not build_subgraph:
                        st.error("build_subgraph() not available.")
                    else:
                        if chosen_uri:
                            try:
                                g_json = build_subgraph(fibo_ttl, chosen_uri, hops=hops)
                                st.session_state["last_fibo_graph"] = g_json
                                cc1, cc2, _ = st.columns([1,1,6])
                                fit_clicked   = cc1.button("Fit", key="btn_fit_graph")
                                reset_clicked = cc2.button("Reset", key="btn_reset_graph")
                                tag = f"{Path(fibo_ttl).name}-{q}-{hops}"
                                components.html(_d3_html(g_json, fit_clicked, reset_clicked, tag), height=620)
                            except Exception as e:
                                st.error(f"Subgraph build failed: {e}")
                        else:
                            st.info("Select a result to render its neighborhood.")
        else:
            st.info("Type a search term to explore FIBO.")
