from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import re

# -------------------- APP --------------------
app = FastAPI(title="Dual Cause–Effect Extraction API")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.run\.app",
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- NLP --------------------
nlp = spacy.load("en_core_web_sm")

# -------------------- INPUT MODEL --------------------
class DualTextInput(BaseModel):
    researcher_analysis: str
    database_reference: str

# -------------------- UTILS --------------------
def clean_line(line: str) -> str:
    line = line.replace('"', '').strip()
    line = re.sub(r"^\s*\d+\.\s*", "", line)
    return line


def extract_cause_effect(sentence: str):
    doc = nlp(sentence)
    cause = None
    effect = None

    root = next((t for t in doc if t.dep_ == "ROOT"), None)

    # Cause (subject)
    if root:
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                cause = " ".join(w.text for w in child.subtree)
                break

    if cause is None and root:
        cause = " ".join(t.text for t in doc if t.i < root.i).strip()

    # Effect (object / complement)
    if root:
        for child in root.children:
            if child.dep_ in ("dobj", "attr", "oprd"):
                effect = " ".join(w.text for w in child.subtree)
                break

    if effect is None:
        for t in doc:
            if t.dep_ in ("xcomp", "ccomp"):
                effect = " ".join(w.text for w in t.subtree)
                break

    return cause, effect


def process_text_block(text: str):
    results = []

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
            "Cause": cause,
            "Effect": effect,
            "SDG": sdg,
            "Polarity": polarity,
            "Directness": directness,
            "Evidence": sentence
        })

    return results

# -------------------- ENDPOINT --------------------
@app.post("/extract-dual")
def extract_dual(data: DualTextInput):
    researcher_results = process_text_block(data.researcher_analysis)
    database_results = process_text_block(data.database_reference)

    return {
        "researcher_analysis_result": researcher_results,
        "database_analysis_result": database_results
    }

# -------------------- HEALTH CHECK --------------------
@app.get("/health")
def health():
    return {"status": "ok"}
