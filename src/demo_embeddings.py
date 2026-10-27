import numpy as np

from embedder import embed

sentences = [
    "The policy covers knee surgery in private hospitals.",
    "Hospitalization expenses are reimbursed up to 5 lakh rupees.",
    "Claims must be filed within 30 days of discharge.",
    "Section 4.2.1 excludes cosmetic procedures.",
    "Our cafeteria serves pizza on Fridays.",
    "The GPU has 24GB of memory.",
]

vectors = embed(sentences)
print("shape:", vectors.shape)
print("length of first vector:", np.linalg.norm(vectors[0]))


def search(query: str, k: int = 3) -> None:
    q = embed([query])[0]
    scores = vectors @ q  
    for i in np.argsort(scores)[::-1][:k]:
        print(f"  {scores[i]:.3f}  {sentences[i]}")


for query in ["Is knee surgery covered?", "How long do I have to submit a claim?", "4.2.1"]:
    print(f"\nQuery: {query}")
    search(query)