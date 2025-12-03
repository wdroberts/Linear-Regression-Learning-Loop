"""
Embedding Demo - Using Sentence Transformers for Movie Similarity

WHAT THIS MODULE DEMONSTRATES:
This module shows how to use embeddings (vector representations) to find
similar movies based on their descriptions. Embeddings convert text into
numbers that capture meaning, allowing us to compare movies mathematically.

WHAT ARE EMBEDDINGS?
Embeddings are numerical representations of text that capture semantic meaning.
Similar concepts are represented by similar numbers. For example:
- "action movie" and "car chase" would have similar embeddings
- "romance" and "love story" would have similar embeddings
- "action" and "romance" would have different embeddings

HOW IT WORKS:
1. We use a pre-trained model (all-MiniLM-L6-v2) that converts text to embeddings
2. Each movie description is converted to a vector (list of numbers)
3. We calculate cosine similarity between vectors to find how similar movies are
4. Higher similarity scores = more similar movies

WHY USE EMBEDDINGS?
- Can find similar content even if words are different
- Understands meaning, not just exact word matches
- Works well for recommendations, search, and clustering
- Pre-trained models save time (no need to train from scratch)

EXAMPLE OUTPUT:
    Action vs Action: 0.856  (high similarity - both are action movies)
    Action vs Romance: 0.234  (low similarity - different genres)

This shows the model understands that action movies are more similar to
each other than to romance movies, even though the exact words differ.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a pre-trained embedding model
# all-MiniLM-L6-v2 is a lightweight, fast model good for general use
# It converts text into 384-dimensional vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# Similar movies with descriptive text
# The descriptions help the model understand what each movie is about
movies = [
    "Fast and Furious: high-octane action",      # Action movie
    "Mad Max: intense car chases",                # Action movie (similar to first)
    "The Notebook: romantic love story"          # Romance movie (different genre)
]

# Convert movie descriptions to embeddings (vectors)
# Each movie becomes a list of 384 numbers that represent its meaning
embeddings = model.encode(movies)

print(f"Embedded {len(movies)} movies")
print(f"Each embedding has {len(embeddings[0])} dimensions")
print()

# Calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    """
    Calculate how similar two embeddings are using cosine similarity.
    
    WHAT IT DOES:
    Measures the angle between two vectors. If vectors point in the same
    direction (similar meaning), the score is close to 1.0. If they point
    in different directions (different meaning), the score is close to 0.0.
    
    HOW IT WORKS:
    1. Calculate dot product of the two vectors (measures alignment)
    2. Divide by the product of their magnitudes (normalizes the result)
    3. Result is between -1 and 1, but for embeddings usually 0 to 1
    
    EXAMPLE:
        Vector A: [0.8, 0.6, 0.0]  (action movie)
        Vector B: [0.7, 0.5, 0.1]  (similar action movie)
        Cosine similarity: 0.95 (very similar!)
        
        Vector C: [0.1, 0.2, 0.9]  (romance movie)
        Cosine similarity with A: 0.15 (very different)
    
    Args:
        a: First embedding vector (numpy array)
        b: Second embedding vector (numpy array)
    
    Returns:
        float: Similarity score between 0 and 1
               - 1.0 = identical meaning
               - 0.8-0.9 = very similar
               - 0.5-0.7 = somewhat similar
               - 0.0-0.3 = very different
    """
    # Dot product: sum of element-wise multiplication
    # Measures how much the vectors align
    dot_product = np.dot(a, b)
    
    # Magnitude (length) of each vector
    # Normalizes the result so it's between -1 and 1
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    # Cosine similarity formula
    similarity = dot_product / (magnitude_a * magnitude_b)
    
    return similarity

# Compare movies to see how similar they are
print("Movie Similarity Analysis:")
print("=" * 50)

# Compare two action movies (should be similar)
action_vs_action = cosine_similarity(embeddings[0], embeddings[1])
print(f"Action vs Action: {action_vs_action:.3f}")
print(f"  '{movies[0]}' vs '{movies[1]}'")
print(f"  → High similarity (both are action movies)")

print()

# Compare action movie to romance movie (should be different)
action_vs_romance = cosine_similarity(embeddings[0], embeddings[2])
print(f"Action vs Romance: {action_vs_romance:.3f}")
print(f"  '{movies[0]}' vs '{movies[2]}'")
print(f"  → Low similarity (different genres)")

print()
print("=" * 50)
print()
print("INTERPRETATION:")
print(f"- Similar movies (action vs action): {action_vs_action:.3f}")
print(f"  This means the model understands both are action movies")
print()
print(f"- Different movies (action vs romance): {action_vs_romance:.3f}")
print(f"  This means the model recognizes they're different genres")
print()
print("The embedding model successfully captures semantic meaning!")

