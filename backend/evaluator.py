from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        print("Chargement du modèle embeddings...")
        _embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Embedder prêt !")
    return _embedder

def compute_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    embedder = get_embedder()
    embeddings = embedder.encode([str(text1), str(text2)])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(sim)

def classify_consistency(score: float):
    if score >= 0.85:
        return "cohérent", "🟢"
    elif score >= 0.65:
        return "partiellement cohérent", "🟡"
    else:
        return "contradictoire", "🔴"