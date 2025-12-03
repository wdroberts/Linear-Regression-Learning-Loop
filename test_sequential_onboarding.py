"""
Test script for sequential onboarding flow
"""
import requests
import json

BASE_URL = "http://localhost:8000"
USER_ID = "test_sequential_user"

print("=" * 60)
print("TESTING SEQUENTIAL ONBOARDING FLOW")
print("=" * 60)

# Test 1: Get first 3 movies
print("\n1. Getting first 3 movies...")
try:
    response = requests.post(f"{BASE_URL}/onboarding/next", json={"user_id": USER_ID})
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Got {len(data['movies'])} movies")
        print(f"   ✓ Selections made: {data['selections_made']}")
        print(f"   ✓ Selections remaining: {data['selections_remaining']}")
        print(f"   ✓ Movies: {[m['name'] for m in data['movies']]}")
        first_round_movies = data['movies']
        first_selection = first_round_movies[0]['name']
    else:
        print(f"   ✗ Error: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("   Make sure the server is running: uvicorn movie_recommendation_lesson6:app --reload")
    exit(1)

# Test 2: Select first movie
print(f"\n2. Selecting first movie: {first_selection}...")
try:
    response = requests.post(
        f"{BASE_URL}/onboarding/select",
        json={"user_id": USER_ID, "movie_name": first_selection}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Status: {data['status']}")
        print(f"   ✓ Selections made: {data['selections_made']}")
        print(f"   ✓ Selections remaining: {data['selections_remaining']}")
        print(f"   ✓ Ready to complete: {data['ready_to_complete']}")
    else:
        print(f"   ✗ Error: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 3: Get second 3 movies
print("\n3. Getting second 3 movies...")
try:
    response = requests.post(f"{BASE_URL}/onboarding/next", json={"user_id": USER_ID})
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Got {len(data['movies'])} movies")
        print(f"   ✓ Selections made: {data['selections_made']}")
        print(f"   ✓ Movies: {[m['name'] for m in data['movies']]}")
        second_round_movies = data['movies']
        second_selection = second_round_movies[0]['name']
    else:
        print(f"   ✗ Error: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 4: Select second movie
print(f"\n4. Selecting second movie: {second_selection}...")
try:
    response = requests.post(
        f"{BASE_URL}/onboarding/select",
        json={"user_id": USER_ID, "movie_name": second_selection}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Selections made: {data['selections_made']}")
        print(f"   ✓ Selections remaining: {data['selections_remaining']}")
    else:
        print(f"   ✗ Error: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 5: Get third 3 movies
print("\n5. Getting third 3 movies...")
try:
    response = requests.post(f"{BASE_URL}/onboarding/next", json={"user_id": USER_ID})
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Got {len(data['movies'])} movies")
        print(f"   ✓ Movies: {[m['name'] for m in data['movies']]}")
        third_round_movies = data['movies']
        third_selection = third_round_movies[0]['name']
    else:
        print(f"   ✗ Error: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 6: Select third movie
print(f"\n6. Selecting third movie: {third_selection}...")
try:
    response = requests.post(
        f"{BASE_URL}/onboarding/select",
        json={"user_id": USER_ID, "movie_name": third_selection}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Selections made: {data['selections_made']}")
        print(f"   ✓ Ready to complete: {data['ready_to_complete']}")
        print(f"   ✓ Message: {data['message']}")
    else:
        print(f"   ✗ Error: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test 7: Complete onboarding
print("\n7. Completing onboarding...")
try:
    response = requests.post(
        f"{BASE_URL}/onboarding/complete",
        json={"user_id": USER_ID}
    )
    if response.status_code == 200:
        data = response.json()
        print(f"   ✓ Status: {data['status']}")
        print(f"   ✓ Initial profile: {data['initial_profile']}")
        print(f"   ✓ Recommended movie: {data['recommendation']['recommended_movie']}")
        print(f"   ✓ Message: {data['message']}")
    else:
        print(f"   ✗ Error: {response.status_code} - {response.text}")
        exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print(f"\nSelected movies: {first_selection}, {second_selection}, {third_selection}")

