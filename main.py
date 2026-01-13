from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import spacy
import re

# ======================================================
# APP
# ======================================================
app = FastAPI(title="IF–IA–SDG Graph Extraction API")

# ======================================================
# CORS (Cloud Run + Google AI Studio)
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.(run\.app|usercontent\.goog)",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# NLP
# ======================================================
nlp = spacy.load("en_core_web_sm")

# ======================================================
# INPUT MODEL (OPTIONAL → NO 422)
# ======================================================
class DualTextInput(BaseModel):
    researcher_analysis: Optional[str] = ""
    database_reference: Optional[str] = ""

# ======================================================
# UTILS
# ======================================================
def clean_line(line: str) -> str:
    line = line.replace('"', '').strip()
    line = re.sub(r"^\s*\d+\.\s*", "", line)
    return line


def extract_cause_effect(sentence: str):
    doc = nlp(sentence)
    cause = None
    effect = None

    root = next((t for t in doc if t.dep_ == "ROOT"), None)

    # Cause
    if root:
        for c in root.children:
            if c.dep_ in ("nsubj", "nsubjpass"):
                cause = " ".join(w.text for w in c.subtree)
                break

    if cause is None and root:
        cause = " ".join(t.text for t in doc if t.i < root.i).strip()

    # Effect
    if root:
        for c in root.children:
            if c.dep_ in ("dobj", "attr", "oprd"):
                effect = " ".join(w.text for w in c.subtree)
                break

    if effect is None:
        for t in doc:
            if t.dep_ in ("xcomp", "ccomp"):
                effect = " ".join(w.text for w in t.subtree)
                break

    return cause, effect


def process_text_block(text: str) -> List[dict]:
    results = []
    if not text:
        return results

    for line in text.split("\n"):
        if not line.strip():
            continue

        line = clean_line(line)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        sentence, sdg, impact_text = parts[:3]
        cause, effect = extract_cause_effect(sentence)

        polarity = (
            "Positive" if "Positive" in impact_text
            else "Negative" if "Negative" in impact_text
            else "More info needed"
        )

        directness = "Direct" if "Direct" in impact_text else "Indirect"

        results.append({
            "Influence_Factor": cause,
            "Influence_Affect": effect,
            "SDG": sdg,
            "Polarity": polarity,
            "Directness": directness,
            "Evidence": sentence
        })

    return results

# ======================================================
# SEMANTIC SIMILARITY
# ======================================================
def semantic_similarity(t1: str, t2: str) -> float:
    if not t1 or not t2:
        return 0.0
    d1, d2 = nlp(t1), nlp(t2)
    if not d1.vector_norm or not d2.vector_norm:
        return 0.0
    return float(d1.similarity(d2))


def similarity_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def compare_researcher_to_database(r_items, d_items):
    comparisons = []

    for r in r_items:
        r_text = f"{r['Influence_Factor']} {r['Influence_Affect']}"
        best_score = 0.0
        best_match = None

        for d in d_items:
            d_text = f"{d['Influence_Factor']} {d['Influence_Affect']}"
            score = semantic_similarity(r_text, d_text)
            if score > best_score:
                best_score = score
                best_match = d

        comparisons.append({
            "researcher_item": r,
            "best_database_match": best_match,
            "similarity_score": round(best_score, 3),
            "alignment": similarity_label(best_score)
        })

    return comparisons

# ======================================================
# GRAPH BUILDER (REACT FLOW READY)
# ======================================================
def build_graph(similarity_results):
    nodes = []
    edges = []
    seen = {}
    idx = 1

    def nid(label, prefix):
        nonlocal idx
        key = f"{prefix}:{label}"
        if key not in seen:
            seen[key] = f"{prefix}_{idx}"
            idx += 1
        return seen[key]

    for item in similarity_results:
        r = item["researcher_item"]

        if_id = nid(r["Influence_Factor"], "IF")
        ia_id = nid(r["Influence_Affect"], "IA")
        sdg_id = nid(r["SDG"], "SDG")

        nodes.extend([
            {"id": if_id, "label": r["Influence_Factor"], "type": "InfluenceFactor"},
            {"id": ia_id, "label": r["Influence_Affect"], "type": "InfluenceAffect"},
            {"id": sdg_id, "label": r["SDG"], "type": "SDG"}
        ])

        is_positive = r["Polarity"] == "Positive"
        is_direct = r["Directness"] == "Direct"

        style = {
            "color": "green" if is_positive else "red",
            "line": "solid" if is_direct else "dashed",
            "strike": not is_positive,
            "width": 3 if item["alignment"] == "high" else 2
        }

        edges.append({
            "source": if_id,
            "target": ia_id,
            "relationship": "causes",
            "polarity": r["Polarity"],
            "directness": r["Directness"],
            "similarity_score": item["similarity_score"],
            "alignment": item["alignment"],
            "style": style
        })

        edges.append({
            "source": ia_id,
            "target": sdg_id,
            "relationship": "impacts"
        })

    return {
        "nodes": list({n["id"]: n for n in nodes}.values()),
        "edges": edges
    }

# ======================================================
# ENDPOINT
# ======================================================
@app.post("/extract-dual")
def extract_dual(data: DualTextInput):
    researcher_results = process_text_block(data.researcher_analysis)
    database_results = process_text_block(data.database_reference)

    similarity_results = compare_researcher_to_database(
        researcher_results,
        database_results
    )

    graph = build_graph(similarity_results)

    return {
        "researcher_analysis_result": researcher_results,
        "database_analysis_result": database_results,
        "semantic_alignment": similarity_results,
        "graph": graph
    }

# ======================================================
# HEALTH
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok"}
