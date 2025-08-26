# app/core/fibo_graph.py
from __future__ import annotations
from pathlib import Path
from rdflib import Graph, RDFS, URIRef
from rdflib.namespace import OWL, SKOS, RDF          # <- add RDF here
from rdflib.util import guess_format

def _label(g, u: URIRef) -> str:
    for p in (RDFS.label, SKOS.prefLabel):
        v = g.value(u, p)
        if v:
            return str(v)
    return str(u).split("/")[-1]

def build_subgraph(
    ttl_path: str | Path,
    focus_uri: str,
    hops: int = 2,
    include_props: bool = True
) -> dict:
    if not focus_uri:
        return {"nodes": [], "links": []}

    ttl_path = Path(ttl_path)
    g = Graph()
    g.parse(ttl_path, format=guess_format(str(ttl_path)) or "turtle")

    # Collect subclass edges
    subclass = [(str(s), str(o)) for s, o in g.subject_objects(RDFS.subClassOf)]

    # Collect object-property domain->range edges (optional)
    props = []
    if include_props:
        for prop, _, _ in g.triples((None, RDF.type, OWL.ObjectProperty)):
            dom = g.value(prop, RDFS.domain)
            rng = g.value(prop, RDFS.range)
            if dom and rng:
                props.append((str(dom), str(rng), str(prop)))

    # Undirected adjacency for neighborhood BFS
    adj: dict[str, set[str]] = {}
    for s, o in subclass:
        adj.setdefault(s, set()).add(o)
        adj.setdefault(o, set()).add(s)
    for d, r, _ in props:
        adj.setdefault(d, set()).add(r)
        adj.setdefault(r, set()).add(d)

    seen = {focus_uri}
    frontier = {focus_uri}
    for _ in range(max(0, hops)):
        nxt = set()
        for u in list(frontier):
            for v in adj.get(u, []):
                if v not in seen:
                    nxt.add(v)
        frontier = nxt
        seen |= frontier

    # Nodes with labels
    def node(u: str) -> dict:
        try:
            return {"id": u, "label": _label(g, URIRef(u))}
        except Exception:
            return {"id": u, "label": u.rsplit("/", 1)[-1]}

    nodes = [node(u) for u in seen]

    # Links (type annotated)
    links = []
    for s, o in subclass:
        if s in seen and o in seen:
            links.append({"source": s, "target": o, "kind": "subClassOf"})
    for d, r, p in props:
        if d in seen and r in seen:
            links.append({"source": d, "target": r, "kind": "property", "label": p})

    return {"nodes": nodes, "links": links}
