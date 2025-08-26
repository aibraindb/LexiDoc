# tools/build_fibo_index.py
import os, json, re
from pathlib import Path
from rdflib import Graph, RDF, RDFS, URIRef
from rdflib.namespace import OWL, SKOS
from rdflib.util import guess_format

DATA = Path("data")
TTL  = DATA / "fibo_full.ttl"
OUT  = DATA / "fibo_index.json"

def _ns(uri: str) -> str:
    m = re.search(r"(https?://.*/ontology/[^/]+/(?:.*/)?)([^/]+)$", uri)
    return m.group(1) if m else uri.rsplit("/",1)[0] + "/"

def _label(g: Graph, u: URIRef) -> str:
    l = g.value(u, RDFS.label) or g.value(u, SKOS.prefLabel)
    return str(l) if l else str(u).split("/")[-1]

def main():
    assert TTL.exists(), f"Missing {TTL}"
    g = Graph()
    fmt = guess_format(str(TTL)) or "turtle"
    g.parse(TTL, format=fmt)

    classes = []
    edges   = []
    ns_counts = {}

    # classes: both OWL.Class and RDFS.Class, plus anything with a label
    seen = set()
    for ctype in (OWL.Class, RDFS.Class):
        for s,_,_ in g.triples((None, RDF.type, ctype)):
            su = str(s)
            if su in seen: continue
            seen.add(su)
            lab = _label(g, s)
            ns = _ns(su)
            ns_counts[ns] = ns_counts.get(ns, 0) + 1
            classes.append({"uri": su, "label": lab, "ns": ns})

    # catch any labeled resources not typed as class
    for s,_,_ in g.triples((None, RDFS.label, None)):
        su = str(s)
        if su in seen: continue
        seen.add(su)
        lab = _label(g, s)
        ns = _ns(su)
        ns_counts[ns] = ns_counts.get(ns, 0) + 1
        classes.append({"uri": su, "label": lab, "ns": ns})

    # subclass edges
    for s,o in g.subject_objects(RDFS.subClassOf):
        edges.append([str(s), str(o)])

    idx = {
        "source": str(TTL),
        "classes": classes,
        "edges": edges,
        "namespaces": [{"ns": ns, "count": ns_counts[ns]} for ns in sorted(ns_counts)],
        "active_ns": []
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(idx))
    print(f"Wrote {OUT} with {len(classes)} classes, {len(edges)} edges.")

if __name__ == "__main__":
    main()
