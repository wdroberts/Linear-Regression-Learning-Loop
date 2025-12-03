"""
Quick test script for onboarding functions
"""
import sys
import numpy as np

# Import the functions from the main module
from movie_recommendation_lesson6 import (
    get_diverse_popular_movies,
    convert_embedding_to_3d,
    onboarding_flow,
    embedding_model,
    MOVIES,
    MOVIE_DESCRIPTIONS,
    init_database
)

print("=" * 60)
print("TESTING ONBOARDING FUNCTIONS")
print("=" * 60)

# Initialize database first
print("\n0. Initializing database...")
try:
    init_database()
    print("   ✓ Database initialized")
except Exception as e:
    print(f"   ⚠ Database init warning: {e}")

# Test 1: get_diverse_popular_movies
print("\n1. Testing get_diverse_popular_movies()...")
try:
    movies = get_diverse_popular_movies(limit=20)
    print(f"   ✓ Returned {len(movies)} movies")
    print(f"   ✓ First movie: {movies[0]['name']} - {movies[0]['description'][:50]}...")
    print(f"   ✓ Sample movie structure: {list(movies[0].keys())}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: convert_embedding_to_3d (if model is available)
print("\n2. Testing convert_embedding_to_3d()...")
if embedding_model is not None:
    try:
        # Create a test embedding from an action movie description
        test_text = "high-octane action thriller with intense sequences"
        test_embedding = embedding_model.encode(test_text)
        result_3d = convert_embedding_to_3d(test_embedding)
        print(f"   ✓ Converted embedding to 3D: {result_3d}")
        print(f"   ✓ Action score: {result_3d[0]:.2f} (should be high for action text)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
else:
    print("   ⚠ Sentence transformer model not available (using fallback)")

# Test 3: onboarding_flow
print("\n3. Testing onboarding_flow()...")
try:
    # Use a test user ID
    test_user_id = "test_user_onboarding"
    # Select 3 action/sci-fi movies
    selected_movies = ["Die Hard", "The Matrix", "Blade Runner"]
    
    print(f"   Testing with user: {test_user_id}")
    print(f"   Selected movies: {selected_movies}")
    
    result = onboarding_flow(test_user_id, selected_movies)
    
    print(f"   ✓ Status: {result['status']}")
    print(f"   ✓ Initial profile: {result['initial_profile']}")
    print(f"   ✓ Message: {result['message']}")
    if result.get('recommendation'):
        print(f"   ✓ Recommended movie: {result['recommendation']['recommended_movie']}")
        print(f"   ✓ Similarity score: {result['recommendation']['similarity_score']}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Verify movie catalog
print("\n4. Verifying movie catalog...")
print(f"   ✓ Total movies in catalog: {len(MOVIES)}")
print(f"   ✓ Movies with descriptions: {len(MOVIE_DESCRIPTIONS)}")
print(f"   ✓ Sample movies: {list(MOVIES.keys())[:5]}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)

