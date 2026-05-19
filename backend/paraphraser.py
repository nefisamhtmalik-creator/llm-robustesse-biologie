import re
import random

synonyms = {
    "est": ["représente", "constitue", "correspond à"],
    "contient": ["renferme", "possède", "comprend"],
    "produit": ["génère", "synthétise", "fabrique"],
    "permet": ["autorise", "rend possible", "facilite"],
    "utilise": ["emploie", "fait appel à"],
    "composé": ["constitué", "formé"],
    "principal": ["essentiel", "fondamental", "primordial"],
    "base": ["fondement", "structure fondamentale"],
    "processus": ["mécanisme", "phénomène", "procédé"],
    "important": ["essentiel", "capital", "fondamental"],
    "spécifique": ["particulier", "propre", "caractéristique"],
}

def lexical_paraphrase(text: str, replacement_rate: float = 0.25) -> str:
    words = text.split()
    new_words = []
    for word in words:
        clean = word.lower().rstrip('.,?!')
        punct = word[len(clean):]
        if clean in synonyms and random.random() < replacement_rate:
            new_words.append(random.choice(synonyms[clean]) + punct)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def syntactic_paraphrase(text: str) -> str:
    transformations = [
        (r"Qu'est-ce que ([^?]+)\s*\?",
         lambda m: f"Quelle est la définition de {m.group(1)} ?"),
        (r"Où se (trouve|trouvent) ([^?]+)\s*\?",
         lambda m: f"Quelle est la localisation de {m.group(2)} ?"),
        (r"Quel est le rôle (de |d')([^?]+)\s*\?",
         lambda m: f"Quelle est la fonction {m.group(1)}{m.group(2)} ?"),
        (r"Quelle est la différence entre ([^e]+) et ([^?]+)\s*\?",
         lambda m: f"Comment distingue-t-on {m.group(1)} de {m.group(2)} ?"),
        (r"Quels? sont les ([^?]+)\s*\?",
         lambda m: f"Énumérez les {m.group(1)}."),
        (r"Comment ([^a]+) agissent?-ils?\s*\?",
         lambda m: f"Quel est le mode d'action de {m.group(1)} ?"),
        (r"Combien de ([^a]+) a ([^?]+)\s*\?",
         lambda m: f"Quel est le nombre de {m.group(1)} que possède {m.group(2)} ?"),
    ]
    for pattern, replacement in transformations:
        result = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        if result != text:
            return result
    return text