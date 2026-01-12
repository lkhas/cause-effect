from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import re

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Cause–Effect Extraction API")

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str


def clean_line(line):
    line = line.replace('"', '').strip()
    line = re.sub(r"^\s*\d+\.\s*", "", line)
    return line


def extract_cause_effect(sentence):
    doc = nlp(sentence)
    cause = None
    effect = None

    root = next((t for t in doc if t.dep_ == "ROOT"), None)

    if root:
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                cause = " ".join(w.text for w in child.subtree)
                break

    if cause is None and root:
        cause = " ".join(t.text for t in doc if t.i < root.i).strip()

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


@app.post("/extract")
def extract(text_input: TextInput):
    results = []

    for line in text_input.text.split("\n"):
        if not line.strip():
            continue

        line = clean_line(line)
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        sentence = parts[0]
        sdg = parts[1]
        impact_text = parts[2]

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

    return {"results": results}
