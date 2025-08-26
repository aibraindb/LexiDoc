from __future__ import annotations
from pathlib import Path
from rdflib import Graph, URIRef, RDFS, RDF, Namespace, Literal
from rdflib.namespace import OWL, SKOS, XSD
from rdflib.util import guess_format
from datetime import datetime
import re, json

CAMEL_RE = re.compile(r'(?<!^)(?=[A-Z])')

def _label(g: Graph, u: URIRef) -> str:
    for p in (RDFS.label, SKOS.prefLabel):
        val = g.value(u, p)
        if val: return str(val)
    return str(u).split("/")[-1]

def _aliases(g: Graph, u: URIRef) -> list[str]:
    alts = set()
    for p in (SKOS.altLabel,):
        for _,_,val in g.triples((u,p,None)):
            try: alts.add(str(val))
            except: pass
    tail = str(u).split("/")[-1]
    if tail:
        alts.add(CAMEL_RE.sub(" ", tail))
    return sorted(a for a in alts if a)

def _datatype(g: Graph, prop: URIRef) -> str | None:
    rng = g.value(prop, RDFS.range)
    if not rng: return None
    s = str(rng)
    # common XSDs
    if s.startswith(str(XSD)):
        return s
    # object property (points to a class) -> treat as ref
    return "ref:" + s

def _superclasses(g: Graph, cls: URIRef) -> set[URIRef]:
    supers = {cls}
    frontier = {cls}
    while frontier:
        nxt = set()
        for c in list(frontier):
            for _,_,o in g.triples((c, RDFS.subClassOf, None)):
                if isinstance(o, URIRef) and o not in supers:
                    supers.add(o); nxt.add(o)
        frontier = nxt
    return supers

def _guess_hints(label: str, dt: str|None) -> list[str]:
    l = label.lower()
    hints = set()
    if dt and dt.startswith(str(XSD.decimal)): hints.add("amount")
    if "amount" in l or "total" in l: hints.add("amount")
    if any(k in l for k in ["date","effective","maturity","signature"]): hints.add("date")
    if any(k in l for k in ["name","party","borrower","lessee","guarantor","lender"]): hints.add("party")
    if "rate" in l: hints.add("rate")
    return sorted(hints)

def build_schema_from_fibo(class_uri: str, ttl_path: str|Path) -> dict:
    ttl_path = Path(ttl_path)
    g = Graph()
    fmt = guess_format(str(ttl_path)) or "turtle"
    g.parse(ttl_path, format=fmt)

    cls = URIRef(class_uri)
    label = _label(g, cls)
    supers = _superclasses(g, cls)

    props = set()
    for p,_,dom in g.triples((None, RDFS.domain, None)):
        if dom in supers:
            props.add(p)

    fields = []
    for p in props:
        plabel = _label(g, p)
        aliases = _aliases(g, p)
        dt = _datatype(g, p)
        fields.append({
            "property_uri": str(p),
            "label": plabel,
            "aliases": aliases,
            "datatype": dt or "unknown",
            "required": False,
            "hints": _guess_hints(plabel, dt),
            "extraction": {"strategy":"auto","window":120}
        })

    fields.sort(key=lambda r: r["label"].lower())
    return {
        "class_uri": class_uri,
        "class_label": label,
        "version": datetime.utcnow().strftime("%Y-%m-%d"),
        "fields": fields,
        "provenance": {
            "fibo_ttl": str(ttl_path),
            "generated_by": "lexigraph-fibo-schema",
            "generated_at": datetime.utcnow().isoformat()+"Z"
        }
    }

def save_schema_json(schema: dict, out_path: str|Path):
    Path(out_path).write_text(json.dumps(schema, indent=2, ensure_ascii=False)):
