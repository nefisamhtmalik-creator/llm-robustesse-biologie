from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sqlite3
from datetime import datetime
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from paraphraser import lexical_paraphrase, syntactic_paraphrase
from evaluator import compute_similarity, classify_consistency
from models import generate_response, AVAILABLE_MODELS

app = FastAPI(
    title="LLM Biology Robustness API",
    description="API pour évaluer la robustesse des LLM en biologie",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'database', 'results.db'
)

def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            paraphrase_syntaxique TEXT,
            paraphrase_lexicale TEXT,
            response_original TEXT,
            response_syntactic TEXT,
            response_lexical TEXT,
            sim_syntactic REAL,
            sim_lexical REAL,
            robustness_score REAL,
            model_used TEXT,
            theme TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("Base de données initialisée !")

init_database()

class QuestionRequest(BaseModel):
    question: str
    model: Optional[str] = "gemini"
    theme: Optional[str] = "général"
    custom_para_syn: Optional[str] = None
    custom_para_lex: Optional[str] = None

class ParaphraseRequest(BaseModel):
    text: str
    paraphrase_type: Optional[str] = "both"

@app.get("/")
def root():
    return {
        "message": "API LLM Biology Robustness",
        "version": "1.0.0",
        "models": AVAILABLE_MODELS
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/paraphrase")
def generate_paraphrase(request: ParaphraseRequest):
    text = request.text
    result = {"original": text}
    if request.paraphrase_type in ["syntaxique", "both"]:
        result["paraphrase_syntaxique"] = syntactic_paraphrase(text)
    if request.paraphrase_type in ["lexicale", "both"]:
        result["paraphrase_lexicale"] = lexical_paraphrase(text)
    return result

@app.post("/evaluate")
def evaluate_question(request: QuestionRequest):
    question = request.question

    # Étape 1 : Paraphrases
    para_syn = request.custom_para_syn if request.custom_para_syn else syntactic_paraphrase(question)
    para_lex = request.custom_para_lex if request.custom_para_lex else lexical_paraphrase(question)

    # Étape 2 : Réponses LLM
    try:
        resp_original  = generate_response(question, request.model)
        resp_syntactic = generate_response(para_syn, request.model)
        resp_lexical   = generate_response(para_lex, request.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur LLM: {str(e)}")

    # Étape 3 : Scores
    sim_syn    = compute_similarity(resp_original, resp_syntactic)
    sim_lex    = compute_similarity(resp_original, resp_lexical)
    robustness = (sim_syn + sim_lex) / 2
    classification, emoji = classify_consistency(robustness)

    # Étape 4 : Sauvegarde
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluations
        (question, paraphrase_syntaxique, paraphrase_lexicale,
         response_original, response_syntactic, response_lexical,
         sim_syntactic, sim_lexical, robustness_score,
         model_used, theme, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (question, para_syn, para_lex,
          resp_original, resp_syntactic, resp_lexical,
          sim_syn, sim_lex, robustness,
          request.model, request.theme,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return {
        "question"              : question,
        "paraphrase_syntaxique" : para_syn,
        "paraphrase_lexicale"   : para_lex,
        "response_original"     : resp_original,
        "response_syntactic"    : resp_syntactic,
        "response_lexical"      : resp_lexical,
        "sim_syntactic"         : round(sim_syn, 4),
        "sim_lexical"           : round(sim_lex, 4),
        "robustness_score"      : round(robustness, 4),
        "classification"        : classification,
        "emoji"                 : emoji,
        "model_used"            : request.model
    }

@app.get("/history")
def get_history(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM evaluations ORDER BY id DESC LIMIT ?',
        (limit,)
    )
    columns = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    return {"count": len(rows), "history": [dict(zip(columns, r)) for r in rows]}

@app.get("/stats")
def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM evaluations')
    total = cursor.fetchone()[0]
    if total == 0:
        return {"message": "Aucune évaluation encore effectuée"}
    cursor.execute('SELECT AVG(robustness_score) FROM evaluations')
    avg_rob = cursor.fetchone()[0]
    cursor.execute('SELECT AVG(sim_syntactic) FROM evaluations')
    avg_syn = cursor.fetchone()[0]
    cursor.execute('SELECT AVG(sim_lexical) FROM evaluations')
    avg_lex = cursor.fetchone()[0]
    conn.close()
    return {
        "total_evaluations"    : total,
        "avg_robustness_score" : round(avg_rob, 4),
        "avg_sim_syntactic"    : round(avg_syn, 4),
        "avg_sim_lexical"      : round(avg_lex, 4),
    }