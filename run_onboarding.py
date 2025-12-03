"""
Interactive Onboarding CLI
Run this script to go through the onboarding flow interactively.
"""
import sys
from movie_recommendation_lesson6 import (
    init_database,
    get_three_movies_for_selection,
    add_movie_selection,
    get_selected_movies,
    get_onboarding_session,
    create_onboarding_session,
    complete_onboarding_session,
    onboarding_flow,
    get_shown_movies
)

def display_movies(movies, selection_number):
    """Display 3 movies in a user-friendly format."""
    print("\n" + "=" * 60)
    print(f"SELECTION {selection_number} OF 3")
    print("=" * 60)
    print("\nChoose one of these movies:\n")
    
    for i, movie in enumerate(movies, 1):
        print(f"{i}. {movie['name']}")
        print(f"   {movie['description']}")
        print()
    
    print("=" * 60)

def get_user_choice():
    """Get user's movie selection (1, 2, or 3)."""
    while True:
        try:
            choice = input("Enter your choice (1, 2, or 3): ").strip()
            if choice in ['1', '2', '3']:
                return int(choice)
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\n\nOnboarding cancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}. Please try again.")

def main():
    """Main onboarding flow."""
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - ONBOARDING")
    print("=" * 60)
    print("\nWelcome! Let's learn about your movie preferences.")
    print("You'll see 3 movies at a time. Pick the one you like most.")
    print("We'll do this 3 times to understand your taste.\n")
    
    # Initialize database
    try:
        init_database()
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)
    
    # Get user ID
    user_id = input("Enter your user ID (or press Enter for 'guest'): ").strip()
    if not user_id:
        user_id = "guest"
    
    print(f"\nStarting onboarding for user: {user_id}\n")
    
    # Create or get session
    if get_onboarding_session(user_id) is None:
        create_onboarding_session(user_id)
    
    # Main selection loop
    for selection_num in range(1, 4):
        # Check if we already have enough selections
        selected = get_selected_movies(user_id)
        if len(selected) >= 3:
            print(f"\nYou've already made 3 selections: {selected}")
            break
        
        # Get 3 movies for this round
        try:
            movies = get_three_movies_for_selection(user_id)
            
            # Mark these movies as shown
            shown_movie_names = [m['name'] for m in movies]
            session = get_onboarding_session(user_id)
            all_shown = session['shown_movies']
            for name in shown_movie_names:
                if name not in all_shown:
                    all_shown.append(name)
            
            # Update shown movies in database
            from movie_recommendation_lesson6 import get_db_connection
            from datetime import datetime
            import json
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE onboarding_sessions SET shown_movies = ?, updated_at = ? WHERE user_id = ?",
                    (json.dumps(all_shown), datetime.now(), user_id)
                )
                conn.commit()
            
            # Display movies
            display_movies(movies, selection_num)
            
            # Get user choice
            choice = get_user_choice()
            selected_movie = movies[choice - 1]['name']
            
            # Record selection
            add_movie_selection(user_id, selected_movie, shown_movie_names)
            
            print(f"\n✓ You selected: {selected_movie}")
            print(f"  Progress: {len(get_selected_movies(user_id))}/3 selections made\n")
            
        except Exception as e:
            print(f"\nError: {e}")
            sys.exit(1)
    
    # Complete onboarding
    selected_movies = get_selected_movies(user_id)
    
    if len(selected_movies) != 3:
        print(f"\nError: Expected 3 selections, but found {len(selected_movies)}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("COMPLETING ONBOARDING...")
    print("=" * 60)
    print(f"\nYour selections: {', '.join(selected_movies)}")
    print("\nCreating your personalized profile...\n")
    
    try:
        result = onboarding_flow(user_id, selected_movies)
        complete_onboarding_session(user_id)
        
        print("=" * 60)
        print("ONBOARDING COMPLETE!")
        print("=" * 60)
        print(f"\n{result['message']}")
        print(f"\nYour profile:")
        profile = result['initial_profile']
        print(f"  Action:    {profile[0]:.1f}/5.0")
        print(f"  Romance:   {profile[1]:.1f}/5.0")
        print(f"  Sci-fi:    {profile[2]:.1f}/5.0")
        
        if result.get('recommendation'):
            rec = result['recommendation']
            print(f"\n🎬 Recommended movie for you: {rec['recommended_movie']}")
            print(f"   (Similarity score: {rec['similarity_score']})")
        
        print("\n" + "=" * 60)
        print("You're all set! Start getting recommendations.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError completing onboarding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

