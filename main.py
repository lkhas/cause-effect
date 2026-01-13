from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import spacy
import re

# ======================================================
# APP
# ======================================================
app = FastAPI(title="Researcher–Database Causal Mapping API")

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
# NLP (use en_core_web_md if available)
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

    # -------- Cause (subject) --------
    if root:
        for c in root.children:
            if c.dep_ in ("nsubj", "nsubjpass"):
                cause = " ".join(w.text for w in c.subtree)
                break

    if cause is None and root:
        cause = " ".join(t.text for t in doc if t.i < root.i).strip()

    # -------- Effect (object / complement) --------
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


def process_text_block(text: str, prefix: str) -> List[dict]:
    results = []
    if not text:
        return results

    for i, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue

        line = clean_line(line)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        sentence, sdg, impact_text = parts[:3]
        cause, effect = extract_cause_effect(sentence)
        
        polarity = (
            "positive" if "positive" in impact_text.lower()
            else "negative" if "negative" in impact_text.lower()
            else "More info needed"
        )
        
        directness = "Direct" if "Direct" in impact_text else "Indirect"

        results.append({
            "id": f"{prefix}{i}",             # R1 / D1 / ...
            "Influence_Factor": cause,
            "Influence_Affect": effect,
            "SDG": sdg,
            "Polarity": polarity,
            "Directness": directness,
            "Evidence": sentence
        })

    return results

# ======================================================
# SEMANTIC SIMILARITY (FULL SENTENCE)
# ======================================================
def semantic_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    d1, d2 = nlp(text1), nlp(text2)
    if not d1.vector_norm or not d2.vector_norm:
        return 0.0
    return float(d1.similarity(d2))


def similarity_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def compare_researcher_to_database(researcher_items, database_items):
    mappings = []

    for r in researcher_items:
        best_score = 0.0
        best_d = None

        for d in database_items:
            score = semantic_similarity(r["Evidence"], d["Evidence"])
            if score > best_score:
                best_score = score
                best_d = d

        mappings.append({
            "mapping_id": f"MAP_{r['id']}_{best_d['id'] if best_d else 'NONE'}",
            "researcher_id": r["id"],
            "database_id": best_d["id"] if best_d else None,
            "similarity_score": round(best_score, 3),
            "alignment": similarity_label(best_score),
            "researcher_sentence": r["Evidence"],
            "database_sentence": best_d["Evidence"] if best_d else None,
            "researcher_item": r,
            "database_item": best_d
        })

    return mappings

# ======================================================
# GRAPH BUILDER (ONE LINE = ONE PATH)
# ======================================================
def build_graph(mappings):
    nodes = []
    edges = []

    for i, m in enumerate(mappings, start=1):
        r = m["researcher_item"]

        if_id = f"IF_{r['id']}"
        ia_id = f"IA_{r['id']}"
        sdg_id = f"SDG_{r['id']}"

        nodes.extend([
            {"id": if_id, "label": r["Influence_Factor"], "type": "InfluenceFactor", "line": r["id"]},
            {"id": ia_id, "label": r["Influence_Affect"], "type": "InfluenceAffect", "line": r["id"]},
            {"id": sdg_id, "label": r["SDG"], "type": "SDG", "line": r["id"]}
        ])

        is_positive = r["Polarity"] == "Positive"
        is_direct = r["Directness"] == "Direct"

        style = {
            "color": "green" if is_positive else "red",
            "line": "solid" if is_direct else "dashed",
            "strike": not is_positive,
            "width": 3 if m["alignment"] == "high" else 2
        }

        edges.append({
            "id": f"E_{r['id']}_IF_IA",
            "source": if_id,
            "target": ia_id,
            "relationship": "causes",
            "polarity": r["Polarity"],
            "directness": r["Directness"],
            "similarity_score": m["similarity_score"],
            "alignment": m["alignment"],
            "style": style
        })

        edges.append({
            "id": f"E_{r['id']}_IA_SDG",
            "source": ia_id,
            "target": sdg_id,
            "relationship": "impacts"
        })

        if m["database_item"]:
            edges.append({
                "id": m["mapping_id"],
                "source": if_id,
                "target": f"IF_{m['database_id']}",
                "relationship": "maps_to",
                "similarity_score": m["similarity_score"],
                "alignment": m["alignment"],
                "style": {
                    "color": "#2563eb",
                    "line": "dashed",
                    "strike": False,
                    "width": 2
                }
            })

    return {
        "nodes": nodes,
        "edges": edges
    }

# ======================================================
# ENDPOINT
# ======================================================
@app.post("/extract-dual")
def extract_dual(data: DualTextInput):
    researcher_results = process_text_block(data.researcher_analysis, "R")
    database_results = process_text_block(data.database_reference, "D")

    mappings = compare_researcher_to_database(
        researcher_results,
        database_results
    )

    graph = build_graph(mappings)

    return {
        "researcher_analysis_result": researcher_results,
        "database_analysis_result": database_results,
        "mappings": mappings,
        "graph": graph
    }

# ======================================================
# HEALTH
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok"}
