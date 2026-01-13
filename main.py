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

        # Case-insensitive polarity detection
        polarity = (
            "positive" if "positive" in impact_text.lower()
            else "negative" if "negative" in impact_text.lower()
            else "More info needed"
        )

        # Case-insensitive directness detection
        impact_lower = impact_text.lower()

        if "indirect" in impact_lower:
            directness = "Indirect"
        elif "direct" in impact_lower:
            directness = "Direct"
        else:
            directness = None


        results.append({
            "id": f"{prefix}{i}",
            "Influence_Factor": cause or "",
            "Influence_Affect": effect or "",
            "SDG": sdg,
            "Polarity": polarity,
            "Directness": directness,
            "Evidence": sentence
        })

    return results

# ======================================================
# ROBUST SEMANTIC SIMILARITY
# ======================================================
def semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    d1, d2 = nlp(text1), nlp(text2)
    if not d1.vector_norm or not d2.vector_norm:
        return 0.0
    return float(d1.similarity(d2))


def calculate_component_similarity(r_item: dict, d_item: dict) -> dict:
    """
    Calculate similarity across multiple components:
    - Influence Factor similarity
    - Influence Affect similarity
    - Full sentence similarity
    - SDG match
    - Polarity match
    - Directness match
    """
    scores = {}
    
    # Component similarities (0-1 range)
    scores['influence_factor_sim'] = semantic_similarity(
        r_item.get("Influence_Factor", ""),
        d_item.get("Influence_Factor", "")
    )
    
    scores['influence_affect_sim'] = semantic_similarity(
        r_item.get("Influence_Affect", ""),
        d_item.get("Influence_Affect", "")
    )
    
    scores['sentence_sim'] = semantic_similarity(
        r_item.get("Evidence", ""),
        d_item.get("Evidence", "")
    )
    
    # Exact matches (binary: 1 or 0)
    scores['sdg_match'] = 1.0 if r_item.get("SDG") == d_item.get("SDG") else 0.0
    
    scores['polarity_match'] = 1.0 if (
        r_item.get("Polarity", "").lower() == d_item.get("Polarity", "").lower()
    ) else 0.0
    
    scores['directness_match'] = 1.0 if (
        r_item.get("Directness", "").lower() == d_item.get("Directness", "").lower()
    ) else 0.0
    
    return scores


def calculate_composite_score(scores: dict, weights: dict = None) -> float:
    """
    Calculate weighted composite similarity score.
    
    Default weights prioritize causal components over metadata:
    - Influence Factor: 30%
    - Influence Affect: 30%
    - Sentence: 20%
    - SDG: 10%
    - Polarity: 5%
    - Directness: 5%
    """
    if weights is None:
        weights = {
            'influence_factor_sim': 0.30,
            'influence_affect_sim': 0.30,
            'sentence_sim': 0.20,
            'sdg_match': 0.10,
            'polarity_match': 0.05,
            'directness_match': 0.05
        }
    
    composite = sum(scores[key] * weights[key] for key in weights.keys())
    return composite


def similarity_label(score: float) -> str:
    """Categorize similarity score into alignment levels."""
    if score >= 0.70:
        return "high"
    if score >= 0.45:
        return "medium"
    if score >= 0.25:
        return "low"
    return "very_low"


def compare_researcher_to_database(researcher_items, database_items, threshold: float = 0.25):
    """
    Enhanced mapping with multi-component similarity.
    
    Args:
        researcher_items: List of researcher analysis items
        database_items: List of database reference items
        threshold: Minimum composite score to consider a match (default: 0.25)
    
    Returns:
        List of mappings with detailed similarity breakdown
    """
    mappings = []

    for r in researcher_items:
        best_score = 0.0
        best_d = None
        best_scores_breakdown = None
        
        # Find all matches above threshold
        all_matches = []

        for d in database_items:
            # Calculate component similarities
            component_scores = calculate_component_similarity(r, d)
            
            # Calculate composite score
            composite_score = calculate_composite_score(component_scores)
            
            if composite_score >= threshold:
                all_matches.append({
                    'database_item': d,
                    'composite_score': composite_score,
                    'component_scores': component_scores
                })
            
            # Track best match
            if composite_score > best_score:
                best_score = composite_score
                best_d = d
                best_scores_breakdown = component_scores

        # Sort matches by composite score
        all_matches.sort(key=lambda x: x['composite_score'], reverse=True)

        mapping = {
            "mapping_id": f"MAP_{r['id']}_{best_d['id'] if best_d else 'NONE'}",
            "researcher_id": r["id"],
            "database_id": best_d["id"] if best_d else None,
            "composite_score": round(best_score, 3),
            "alignment": similarity_label(best_score),
            "researcher_sentence": r["Evidence"],
            "database_sentence": best_d["Evidence"] if best_d else None,
            "researcher_item": r,
            "database_item": best_d,
            "similarity_breakdown": {
                "influence_factor": round(best_scores_breakdown.get('influence_factor_sim', 0), 3) if best_scores_breakdown else 0,
                "influence_affect": round(best_scores_breakdown.get('influence_affect_sim', 0), 3) if best_scores_breakdown else 0,
                "sentence": round(best_scores_breakdown.get('sentence_sim', 0), 3) if best_scores_breakdown else 0,
                "sdg_match": bool(best_scores_breakdown.get('sdg_match', 0)) if best_scores_breakdown else False,
                "polarity_match": bool(best_scores_breakdown.get('polarity_match', 0)) if best_scores_breakdown else False,
                "directness_match": bool(best_scores_breakdown.get('directness_match', 0)) if best_scores_breakdown else False
            },
            "alternative_matches": [
                {
                    "database_id": m['database_item']['id'],
                    "composite_score": round(m['composite_score'], 3),
                    "alignment": similarity_label(m['composite_score'])
                }
                for m in all_matches[1:4]  # Include top 3 alternatives
            ] if len(all_matches) > 1 else []
        }
        
        mappings.append(mapping)

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

        # Case-insensitive comparison
        is_positive = r["Polarity"].lower() == "positive"
        is_direct = r["Directness"].lower() == "direct"

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
            "composite_score": m["composite_score"],
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
            # Adjust mapping edge width based on composite score
            mapping_width = 3 if m["composite_score"] >= 0.70 else 2 if m["composite_score"] >= 0.45 else 1
            
            edges.append({
                "id": m["mapping_id"],
                "source": if_id,
                "target": f"IF_{m['database_id']}",
                "relationship": "maps_to",
                "composite_score": m["composite_score"],
                "alignment": m["alignment"],
                "similarity_breakdown": m["similarity_breakdown"],
                "style": {
                    "color": "#2563eb",
                    "line": "dashed",
                    "strike": False,
                    "width": mapping_width
                }
            })

    return {
        "nodes": nodes,
        "edges": edges
    }

# ======================================================
# SDG SORTING UTILITY
# ======================================================
def extract_sdg_number(sdg_string: str) -> int:
    """Extract numeric value from SDG string (e.g., 'SDG 3' -> 3)"""
    if not sdg_string:
        return 999  # Put items without SDG at the end
    
    # Extract first number found in the string
    match = re.search(r'\d+', str(sdg_string))
    return int(match.group()) if match else 999

def sort_by_sdg(items: List[dict]) -> List[dict]:
    """Sort items by SDG number"""
    return sorted(items, key=lambda x: extract_sdg_number(x.get("SDG", "")))

# ======================================================
# ENDPOINT
# ======================================================
@app.post("/extract-dual")
def extract_dual(data: DualTextInput):
    researcher_results = process_text_block(data.researcher_analysis, "R")
    database_results = process_text_block(data.database_reference, "D")
    
    # Sort both results by SDG number
    researcher_results = sort_by_sdg(researcher_results)
    database_results = sort_by_sdg(database_results)

    mappings = compare_researcher_to_database(
        researcher_results,
        database_results
    )

    graph = build_graph(mappings)

    return {
        "researcher_analysis_result": researcher_results,
        "database_analysis_result": database_results,
        "mappings": mappings,
        "graph": graph,
        "statistics": {
            "total_researcher_items": len(researcher_results),
            "total_database_items": len(database_results),
            "high_confidence_matches": sum(1 for m in mappings if m["alignment"] == "high"),
            "medium_confidence_matches": sum(1 for m in mappings if m["alignment"] == "medium"),
            "low_confidence_matches": sum(1 for m in mappings if m["alignment"] == "low"),
            "no_matches": sum(1 for m in mappings if m["database_item"] is None)
        }
    }

# ======================================================
# HEALTH
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok"}