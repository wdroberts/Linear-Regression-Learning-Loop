# Movie Recommendation Learning Loop - PRODUCTION-LITE VERSION
# This demonstrates how the simple learning loop would look in a production environment
# 
# NEW PRODUCTION FEATURES:
# 1. FastAPI endpoints for receiving feedback and serving recommendations
# 2. SQLite database for persistent storage of profiles and feedback
# 3. Automatic retraining triggered after collecting N feedback items
# 4. Logging for monitoring system behavior
# 5. Separation of concerns (API layer, business logic, data layer)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Iterator
from contextlib import contextmanager

# ============================================================================
# SECTION 1: LOGGING SETUP (Production Monitoring)
# ============================================================================
# Purpose: Track what's happening in the system for debugging and monitoring
# Why: In production, you need to know when retraining happens, errors occur, etc.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 2: DATA MODELS (API Request/Response Schemas)
# ============================================================================
# Purpose: Define the structure of data coming in and going out of API endpoints
# Why: FastAPI uses these to validate requests and auto-generate documentation

class FeedbackRequest(BaseModel):
    """Schema for submitting user feedback"""
    user_id: str
    movie_name: str
    rating: int  # 1 = thumbs up, 0 = thumbs down

class RecommendationResponse(BaseModel):
    """Schema for recommendation responses"""
    user_id: str
    recommended_movie: str
    similarity_score: float
    user_profile: List[float]
    all_scores: Dict[str, float]

# ============================================================================
# SECTION 3: DATABASE SETUP & INITIALIZATION
# ============================================================================
# Purpose: Create persistent storage for user profiles and feedback
# Why: Data must survive restarts (unlike in-memory lists)

DATABASE_PATH = "movie_recommendations.db"

# ============================================================================
# DATABASE CONTEXT MANAGER (Safe Connection Handling)
# ============================================================================
# Purpose: Ensure database connections are always properly closed
# Why: Prevents connection leaks and ensures cleanup even on errors

@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """
    Context manager for safely handling database connections.
    
    WHAT IT DOES:
    This is a special Python function that ensures database connections are always
    properly closed, even if an error occurs. Think of it like a safety net.
    
    HOW IT WORKS:
    1. Opens a connection to the SQLite database
    2. Lets you use the connection (via 'yield')
    3. If everything succeeds: saves (commits) all changes
    4. If an error occurs: undoes (rolls back) all changes
    5. Always closes the connection when done
    
    WHY WE NEED IT:
    Without this, if an error happens, the database connection might stay open,
    which can cause problems. This function guarantees cleanup.
    
    USAGE EXAMPLE:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            # Connection automatically closes here, even if error occurs
    
    Returns:
        Iterator[sqlite3.Connection]: A context manager that yields a SQLite
                                      database connection object. The connection
                                      is automatically closed when exiting the
                                      'with' block, even if an error occurs.
    
    Type Hints:
        The return type `Iterator[sqlite3.Connection]` indicates this function
        yields (produces) a SQLite connection object. This helps type checkers
        and IDEs understand what type of object you'll get when using this
        context manager.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn  # Give the connection to the code using 'with'
        conn.commit()  # Save all changes if no errors occurred
    except Exception as e:
        conn.rollback()  # Undo all changes if an error occurred
        logger.error(f"Database error: {e}")
        raise  # Re-raise the error so calling code knows something went wrong
    finally:
        conn.close()  # Always close the connection, no matter what

def init_database() -> None:
    """
    Create the database and all required tables if they don't already exist.
    
    WHAT IT DOES:
    This function sets up the database structure. It's like building the foundation
    of a house - it creates the "rooms" (tables) where we'll store data.
    
    HOW IT WORKS:
    1. Connects to the database (creates it if it doesn't exist)
    2. Creates the 'user_profiles' table to store user preferences
    3. Creates the 'feedback' table to store movie ratings
    4. Uses "IF NOT EXISTS" so it won't error if tables already exist
    
    TABLES CREATED:
    - user_profiles: Stores each user's preference vector (embedding)
      * user_id: Unique identifier for each user (like a username)
      * embedding: JSON string of the user's preferences [action, romance, sci-fi]
      * last_updated: When the profile was last modified
    
    - feedback: Stores all movie ratings from users
      * id: Auto-incrementing unique ID for each rating
      * user_id: Which user gave this rating
      * movie_name: Which movie was rated
      * rating: 1 = liked, 0 = disliked
      * timestamp: When the rating was given
      * processed: 0 = not used for retraining yet, 1 = already used
    
    WHEN IT'S CALLED:
    Automatically called when the API server starts up.
    
    Returns:
        None: This function doesn't return a value. It creates database
              tables as a side effect. The `-> None` type hint explicitly
              indicates that this function performs actions but doesn't
              return any data.
    
    Side Effects:
        - Creates the SQLite database file if it doesn't exist
        - Creates 'user_profiles' table if it doesn't exist
        - Creates 'feedback' table if it doesn't exist
        - Logs database initialization status
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Table for user profiles (embeddings)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Table for feedback (ratings)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                movie_name TEXT NOT NULL,
                rating INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed INTEGER DEFAULT 0
            )
        ''')
    
    logger.info("Database initialized successfully")

# ============================================================================
# SECTION 4: MOVIE CATALOG (In-Memory for Simplicity)
# ============================================================================
# Purpose: Store movie embeddings
# Note: In real production, this would also be in a database table

MOVIES = {
    "Die Hard": [5, 1, 0],
    "The Notebook": [0, 5, 1],
    "Blade Runner": [3, 0, 5],
    "Titanic": [2, 4, 0],
    "The Matrix": [4, 1, 4]
}

# ============================================================================
# RETRAINING CONFIGURATION
# ============================================================================
# These constants control when and how the system retrains user profiles

# RETRAIN_THRESHOLD: How many unprocessed feedback items trigger retraining
# 
# WHAT IT DOES:
# This is the "trigger" - when a user has given this many new ratings that
# haven't been used for retraining yet, the system will automatically retrain
# their profile.
#
# EXAMPLE:
#   RETRAIN_THRESHOLD = 3
#   User gives 1st rating → No retraining (1 < 3)
#   User gives 2nd rating → No retraining (2 < 3)
#   User gives 3rd rating → RETRAINING TRIGGERED! (3 >= 3)
#
# WHY NOT RETRAIN AFTER EVERY RATING?
# - More efficient: batch processing is faster than one-by-one
# - More stable: prevents overreacting to a single rating
# - Better patterns: multiple ratings reveal clearer preferences
RETRAIN_THRESHOLD = 3  # Retrain after every 3 new ratings

# SLIDING_WINDOW_DAYS: Only use feedback from the last N days for retraining
#
# WHAT IS A SLIDING WINDOW?
# A "sliding window" is like a moving time frame. We only look at feedback
# from the last N days. As time passes, old feedback "slides out" of the
# window and is ignored. Think of it like a camera that only sees the
# last 90 days - everything older is out of view.
#
# WHY USE A SLIDING WINDOW?
# User preferences change over time! A movie they loved 6 months ago might
# not reflect their current tastes. By only using recent feedback:
# - The model adapts to changing preferences
# - Old, irrelevant feedback doesn't affect current recommendations
# - The system stays current with the user's evolving tastes
#
# EXAMPLE:
#   SLIDING_WINDOW_DAYS = 90
#   
#   Today is January 1st:
#   - Feedback from December 15th (17 days ago): INCLUDED ✓ (within 90 days)
#   - Feedback from October 1st (92 days ago): EXCLUDED ✗ (outside window)
#   - Feedback from 6 months ago: EXCLUDED ✗ (too old)
#
#   On February 1st (31 days later):
#   - Feedback from December 15th: Still INCLUDED ✓ (now 48 days ago, still < 90)
#   - Feedback from October 1st: Still EXCLUDED ✗ (now 123 days ago, > 90)
#
# HOW IT WORKS:
# When retraining, the system:
# 1. Calculates cutoff date: today - SLIDING_WINDOW_DAYS
# 2. Only uses feedback with timestamp >= cutoff date
# 3. Ignores all older feedback (even if it's unprocessed)
#
# TUNING THIS VALUE:
# - Smaller (30-60 days): More responsive to recent changes, but less stable
# - Larger (120-180 days): More stable, but slower to adapt to preference changes
# - Default (90 days): Good balance for most use cases
SLIDING_WINDOW_DAYS = 90  # Only use feedback from last 90 days for retraining

# MIN_FEEDBACK_FOR_RETRAIN: Minimum recent feedback needed to retrain
#
# WHAT IT DOES:
# This ensures we have enough recent feedback (within the sliding window)
# before retraining. Even if we have RETRAIN_THRESHOLD unprocessed items,
# we won't retrain unless at least MIN_FEEDBACK_FOR_RETRAIN of them are
# within the sliding window.
#
# WHY WE NEED BOTH THRESHOLDS:
# - RETRAIN_THRESHOLD: Checks total unprocessed feedback (could be old)
# - MIN_FEEDBACK_FOR_RETRAIN: Checks recent feedback (within sliding window)
#
# EXAMPLE SCENARIO:
#   RETRAIN_THRESHOLD = 3
#   MIN_FEEDBACK_FOR_RETRAIN = 3
#   SLIDING_WINDOW_DAYS = 90
#
#   User has:
#   - 5 unprocessed ratings from 120 days ago (outside window)
#   - 2 unprocessed ratings from last week (inside window)
#
#   Result:
#   - Total unprocessed = 7 (>= RETRAIN_THRESHOLD of 3) ✓
#   - Recent unprocessed = 2 (< MIN_FEEDBACK_FOR_RETRAIN of 3) ✗
#   - RETRAINING NOT TRIGGERED (need at least 3 recent ratings)
#
#   After user gives 1 more recent rating:
#   - Recent unprocessed = 3 (>= MIN_FEEDBACK_FOR_RETRAIN) ✓
#   - RETRAINING TRIGGERED! (uses only the 3 recent ratings)
#
# RELATIONSHIP TO RETRAIN_THRESHOLD:
# MIN_FEEDBACK_FOR_RETRAIN should typically be <= RETRAIN_THRESHOLD.
# If it's greater, you might never trigger retraining even with enough
# unprocessed feedback (because recent feedback requirement is too high).
MIN_FEEDBACK_FOR_RETRAIN = 3  # Minimum recent feedback needed to retrain

# ============================================================================
# SECTION 5: CORE BUSINESS LOGIC (Same as Before, Now Functions)
# ============================================================================

def calculate_similarity(profile: List[float], movie_embedding: List[float]) -> float:
    """
    Calculate how well a movie matches a user's preferences.
    
    WHAT IT DOES:
    Compares the user's preference profile to a movie's characteristics and
    calculates a "distance" score. Lower scores mean better matches.
    
    HOW IT WORKS (Step by step):
    1. Takes two lists of numbers (vectors):
       - profile: User's preferences [action, romance, sci-fi]
       - movie_embedding: Movie's characteristics [action, romance, sci-fi]
    
    2. For each dimension (action, romance, sci-fi):
       - Calculates the difference: user_preference - movie_characteristic
       - Takes the absolute value (makes it positive)
       - Adds it to a running total
    
    3. Returns the total distance (lower = better match)
    
    EXAMPLE:
        User profile: [5, 2, 1]  (loves action, some romance, little sci-fi)
        Movie:        [4, 1, 0]  (high action, low romance, no sci-fi)
        
        Action difference:  |5 - 4| = 1
        Romance difference: |2 - 1| = 1
        Sci-fi difference:  |1 - 0| = 1
        Total similarity score: 1 + 1 + 1 = 3 (good match!)
        
        If movie was [0, 5, 1] (no action, high romance):
        Action: |5 - 0| = 5
        Romance: |2 - 5| = 3
        Sci-fi: |1 - 1| = 0
        Total: 5 + 3 + 0 = 8 (worse match)
    
    WHY LOWER IS BETTER:
    Think of it like GPS distance - if you're at point A and want to get to point B,
    a smaller distance means you're closer. Same here - smaller distance = closer match.
    
    Args:
        profile: List of 3 numbers representing user preferences [action, romance, sci-fi]
                Each number is 0-5, where 5 = maximum preference
        movie_embedding: List of 3 numbers representing movie characteristics
                        Same format: [action, romance, sci-fi]
    
    Returns:
        float: A similarity score (distance). Lower values = better matches.
              Typical range: 0-15 (0 = perfect match, 15 = completely opposite)
    """
    total = 0  # Start with zero distance
    
    # Compare each dimension (action, romance, sci-fi)
    for i in range(len(profile)):
        # Calculate difference in this dimension
        diff = profile[i] - movie_embedding[i]
        # Use absolute value (make it positive) and add to total
        total += abs(diff)
    
    return total  # Return the total distance

def retrain_profile(current_profile: List[float], feedback_data: List[Dict]) -> List[float]:
    """
    Update the user's preference profile based on their movie ratings.
    
    WHAT IT DOES:
    This is the "learning" part of the system. When a user rates movies, we adjust
    their profile to better reflect their actual preferences.
    
    HOW IT WORKS:
    1. Starts with a copy of the current profile (so we don't modify the original)
    2. For each movie the user rated:
       - If they LIKED it (rating=1): Move profile TOWARD that movie's characteristics
       - If they DISLIKED it (rating=0): Move profile AWAY from that movie's characteristics
    3. Uses a learning rate of 0.3 (30%) - this means we only move 30% of the way
       toward/away from each movie. This prevents the profile from changing too drastically.
    
    EXAMPLE - User LIKES an action movie:
        Current profile: [3.0, 3.0, 3.0]  (balanced preferences)
        Movie "Die Hard": [5, 1, 0]  (high action, low romance, no sci-fi)
        User rating: 1 (liked it)
        
        For ACTION dimension:
        - Difference: 5 - 3.0 = 2.0
        - Move 30%: 2.0 * 0.3 = 0.6
        - New action preference: 3.0 + 0.6 = 3.6 (moved toward action)
        
        For ROMANCE dimension:
        - Difference: 1 - 3.0 = -2.0
        - Move 30%: -2.0 * 0.3 = -0.6
        - New romance preference: 3.0 + (-0.6) = 2.4 (moved away from romance)
        
        Result: Profile becomes [3.6, 2.4, 2.7] - more action-focused!
    
    EXAMPLE - User DISLIKES a romance movie:
        Current profile: [3.0, 3.0, 3.0]
        Movie "The Notebook": [0, 5, 1]  (no action, high romance)
        User rating: 0 (disliked it)
        
        For ACTION dimension:
        - Difference: 0 - 3.0 = -3.0
        - Move 30% AWAY: -(-3.0) * 0.3 = 0.9
        - New action: 3.0 + 0.9 = 3.9 (moved away from "no action")
        
        Result: Profile moves away from romance-heavy movies.
    
    WHY LEARNING RATE MATTERS:
    The 0.3 (30%) learning rate means we don't completely change the profile based
    on one rating. This makes the system more stable and prevents overreacting to
    a single movie. It's like learning gradually rather than changing your mind
    completely after one experience.
    
    Args:
        current_profile: List of 3 floats representing current user preferences
                        Format: [action_preference, romance_preference, sci-fi_preference]
                        Each value is typically 0.0 to 5.0
        feedback_data: List of dictionaries, each containing:
                     - 'movie_name': str - Name of the movie rated
                     - 'rating': int - 1 if liked, 0 if disliked
                     Example: [{'movie_name': 'Die Hard', 'rating': 1}, ...]
    
    Returns:
        List[float]: Updated profile with adjusted preferences
                   Same format as input: [action, romance, sci-fi]
                   Values may have changed based on feedback
    
    Note:
        This function does NOT save to database - that's done separately by
        calling save_user_profile() after this function.
    """
    # Make a copy so we don't modify the original profile
    new_profile = current_profile.copy()
    
    # Process each piece of feedback
    for item in feedback_data:
        movie_name = item['movie_name']
        rating = item['rating']  # 1 = liked, 0 = disliked
        
        # Safety check: skip if movie not in our catalog
        if movie_name not in MOVIES:
            logger.warning(f"Movie {movie_name} not in catalog, skipping")
            continue
        
        # Get the movie's characteristics
        movie_embedding = MOVIES[movie_name]
        
        # Adjust each dimension of the profile
        for i in range(len(new_profile)):
            if rating == 1:  # User LIKED this movie
                # Move profile TOWARD this movie's characteristics
                # Formula: current + (difference * learning_rate)
                difference = movie_embedding[i] - new_profile[i]
                adjustment = difference * 0.3  # 30% learning rate
                new_profile[i] += adjustment
            else:  # User DISLIKED this movie (rating = 0)
                # Move profile AWAY from this movie's characteristics
                # Formula: current - (difference * learning_rate)
                difference = movie_embedding[i] - new_profile[i]
                adjustment = difference * 0.3  # 30% learning rate
                new_profile[i] -= adjustment
    
    return new_profile

# ============================================================================
# SECTION 6: DATABASE OPERATIONS (Data Layer)
# ============================================================================
# Purpose: Separate database access from business logic
# Why: Makes code testable and easier to modify (e.g., switch to PostgreSQL)

def get_user_profile(user_id: str) -> List[float]:
    """
    Get a user's preference profile from the database, or create one if they're new.
    
    WHAT IT DOES:
    Retrieves a user's movie preferences (their "profile") from the database.
    If the user doesn't exist yet, it creates a new profile with balanced preferences.
    
    HOW IT WORKS:
    1. Connects to the database
    2. Looks up the user by their user_id
    3. If found: Retrieves and decodes their stored profile
    4. If not found: Creates a new profile with balanced preferences [3.0, 3.0, 3.0]
       and saves it to the database
    
    WHAT IS A PROFILE?
    A profile is a list of 3 numbers representing how much a user likes:
    - Action movies (index 0)
    - Romance movies (index 1)
    - Sci-fi movies (index 2)
    
    Each number ranges from 0.0 to 5.0:
    - 0.0 = doesn't like at all
    - 3.0 = neutral/balanced
    - 5.0 = loves this genre
    
    EXAMPLE:
        get_user_profile("sarah")
        # Returns: [5.0, 2.0, 1.0]
        # Meaning: Sarah loves action (5), likes some romance (2), not much sci-fi (1)
    
    DEFAULT PROFILE:
    New users start with [3.0, 3.0, 3.0] - balanced preferences. As they rate movies,
    their profile will adjust to reflect their actual tastes.
    
    Args:
        user_id: A string identifying the user (e.g., "sarah", "john123")
                This is like a username or unique identifier
    
    Returns:
        List[float]: The user's preference profile as [action, romance, sci-fi]
                   Always returns exactly 3 numbers, each between 0.0 and 5.0
    
    Side Effects:
        If user doesn't exist, creates a new database record with default profile
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT embedding FROM user_profiles WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
        
        if result:
            profile = json.loads(result[0])
            logger.info(f"Retrieved profile for user {user_id}")
        else:
            # Create default profile: balanced preferences
            profile = [3.0, 3.0, 3.0]  # [action, romance, sci-fi]
            cursor.execute(
                "INSERT INTO user_profiles (user_id, embedding) VALUES (?, ?)",
                (user_id, json.dumps(profile))
            )
            logger.info(f"Created default profile for new user {user_id}")
    
    return profile

def save_user_profile(user_id: str, profile: List[float]):
    """
    Save an updated user profile to the database.
    
    WHAT IT DOES:
    Stores a user's preference profile in the database, updating it if it already exists.
    This is called after retraining to save the learned preferences.
    
    HOW IT WORKS:
    1. Connects to the database
    2. Converts the profile list to JSON format (databases store text, not lists)
    3. Updates the user's record with the new profile
    4. Updates the last_updated timestamp to track when it was modified
    
    WHEN IT'S CALLED:
    - After retraining (when user profile is updated based on feedback)
    - When a new user is created (though get_user_profile() handles that)
    
    EXAMPLE:
        profile = [4.2, 2.1, 1.8]  # Updated after learning from feedback
        save_user_profile("sarah", profile)
        # Saves this profile to database for user "sarah"
    
    Args:
        user_id: String identifying which user to update
        profile: List of 3 floats [action, romance, sci-fi] to save
    
    Returns:
        None (just saves to database, doesn't return anything)
    
    Note:
        This will overwrite any existing profile for this user.
        Make sure the profile is the complete updated version.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE user_profiles SET embedding = ?, last_updated = ? WHERE user_id = ?",
            (json.dumps(profile), datetime.now(), user_id)
        )
    logger.info(f"Saved updated profile for user {user_id}")

def save_feedback(user_id: str, movie_name: str, rating: int):
    """
    Save a user's movie rating to the database.
    
    WHAT IT DOES:
    Stores a single movie rating from a user. The rating is marked as "unprocessed"
    (processed=0) so it can be used later for retraining.
    
    HOW IT WORKS:
    1. Connects to the database
    2. Inserts a new record into the feedback table with:
       - user_id: Who gave this rating
       - movie_name: Which movie was rated
       - rating: 1 (liked) or 0 (disliked)
       - processed: 0 (not yet used for retraining)
       - timestamp: Automatically set to current time
    
    WHAT HAPPENS NEXT:
    After saving, the system checks if enough feedback has accumulated to trigger
    retraining. Once feedback is used for retraining, it's marked as processed=1.
    
    EXAMPLE:
        save_feedback("sarah", "Die Hard", 1)
        # Saves: Sarah liked Die Hard
        # This feedback will be used later to update Sarah's profile
    
    Args:
        user_id: String identifying the user who gave the rating
        movie_name: Name of the movie being rated (must exist in MOVIES catalog)
        rating: Integer - 1 means "liked" (thumbs up), 0 means "disliked" (thumbs down)
    
    Returns:
        None (just saves to database)
    
    Note:
        The feedback is stored but not immediately used. It will be processed
        when retraining is triggered (after RETRAIN_THRESHOLD feedback items).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (user_id, movie_name, rating, processed) VALUES (?, ?, ?, 0)",
            (user_id, movie_name, rating)
        )
    logger.info(f"Saved feedback: {user_id} rated {movie_name} as {'👍' if rating == 1 else '👎'}")

def get_recent_feedback(user_id: str, days: int = SLIDING_WINDOW_DAYS) -> List[Dict]:
    """
    Get feedback from a user within the last N days (sliding window).
    
    WHAT IT DOES:
    Retrieves feedback (ratings) for a specific user that are within a time window.
    This implements a "sliding window" approach - only recent feedback is used
    for retraining, ensuring the model adapts to changing user preferences.
    
    HOW IT WORKS:
    1. Calculates a cutoff date (now - N days)
    2. Queries for feedback where:
       - user_id matches the requested user
       - timestamp is within the window (>= cutoff_date)
    3. Returns a list of dictionaries with movie name, rating, and timestamp info
    
    WHY SLIDING WINDOW?
    - User preferences change over time
    - Old feedback may not reflect current tastes
    - Only using recent feedback keeps the model current
    - Prevents stale data from affecting recommendations
    
    EXAMPLE:
        get_recent_feedback("sarah", days=90)
        # Returns only feedback from last 90 days:
        # [
        #     {'movie_name': 'Die Hard', 'rating': 1, 'days_ago': 5},
        #     {'movie_name': 'The Matrix', 'rating': 1, 'days_ago': 12},
        #     {'movie_name': 'The Notebook', 'rating': 0, 'days_ago': 45}
        # ]
        # Feedback older than 90 days is excluded
    
    Args:
        user_id: String identifying which user's feedback to retrieve
        days: Number of days to look back (default: SLIDING_WINDOW_DAYS = 90)
             Only feedback from the last N days will be included
    
    Returns:
        List[Dict]: List of dictionaries, each with:
                   - 'movie_name': str - Name of the movie
                   - 'rating': int - 1 (liked) or 0 (disliked)
                   - 'days_ago': int - How many days ago the feedback was given
                   - 'timestamp': datetime - When the feedback was recorded
                   Returns empty list [] if no recent feedback exists
    
    Note:
        This function returns ALL recent feedback, regardless of processed status.
        For retraining, we typically use unprocessed feedback within this window.
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT movie_name, rating, timestamp 
            FROM feedback 
            WHERE user_id = ? AND timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (user_id, cutoff_date.isoformat())
        )
        results = cursor.fetchall()
    
    feedback_list = []
    for row in results:
        movie_name, rating, timestamp_str = row
        timestamp = datetime.fromisoformat(timestamp_str) if isinstance(timestamp_str, str) else timestamp_str
        days_ago = (datetime.now() - timestamp).days
        
        feedback_list.append({
            'movie_name': movie_name,
            'rating': rating,
            'days_ago': days_ago,
            'timestamp': timestamp
        })
    
    return feedback_list

def get_unprocessed_feedback(user_id: str) -> List[Dict]:
    """
    Get all movie ratings from a user that haven't been used for retraining yet.
    
    WHAT IT DOES:
    Retrieves all feedback (ratings) for a specific user that are marked as
    "unprocessed" (processed=0). These are ratings that haven't been used to
    update the user's profile yet.
    
    HOW IT WORKS:
    1. Connects to the database
    2. Queries for all feedback where:
       - user_id matches the requested user
       - processed = 0 (not yet used for retraining)
    3. Returns a list of dictionaries, each containing the movie name and rating
    
    WHEN IT'S CALLED:
    Called by check_and_retrain() to get all the new feedback that should be
    used to update the user's profile.
    
    EXAMPLE:
        get_unprocessed_feedback("sarah")
        # Returns: [
        #     {'movie_name': 'Die Hard', 'rating': 1},
        #     {'movie_name': 'The Matrix', 'rating': 1},
        #     {'movie_name': 'The Notebook', 'rating': 0}
        # ]
        # These are all the ratings Sarah gave that haven't been processed yet
    
    Args:
        user_id: String identifying which user's feedback to retrieve
    
    Returns:
        List[Dict]: List of dictionaries, each with:
                   - 'movie_name': str - Name of the movie
                   - 'rating': int - 1 (liked) or 0 (disliked)
                   Returns empty list [] if no unprocessed feedback exists
    
    Note:
        After retraining, these feedback items should be marked as processed
        using mark_feedback_processed() so they aren't used again.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT movie_name, rating FROM feedback WHERE user_id = ? AND processed = 0",
            (user_id,)
        )
        results = cursor.fetchall()
    
    return [{'movie_name': row[0], 'rating': row[1]} for row in results]

def mark_feedback_processed(user_id: str):
    """
    Mark all unprocessed feedback for a user as "processed" (already used for retraining).
    
    WHAT IT DOES:
    Updates all feedback records for a user, changing processed from 0 to 1.
    This prevents the same feedback from being used multiple times for retraining.
    
    HOW IT WORKS:
    1. Connects to the database
    2. Finds all feedback for this user where processed = 0
    3. Updates them all to processed = 1
    
    WHY WE NEED THIS:
    After retraining uses feedback to update a profile, we mark it as processed.
    This ensures:
    - We don't use the same feedback twice (which would over-adjust the profile)
    - We can track which feedback has been "learned from"
    - We know how much new feedback is waiting to be processed
    
    WHEN IT'S CALLED:
    Called automatically by check_and_retrain() after successfully updating
    the user's profile based on their feedback.
    
    EXAMPLE:
        # Before: Sarah has 3 unprocessed ratings
        # After retraining and calling this function:
        mark_feedback_processed("sarah")
        # Now all 3 ratings are marked as processed=1
        # They won't be used again for retraining
    
    Args:
        user_id: String identifying which user's feedback to mark as processed
    
    Returns:
        None (just updates database records)
    
    Note:
        This is a one-way operation. Once feedback is marked as processed,
        it stays processed (we don't un-process it).
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE feedback SET processed = 1 WHERE user_id = ? AND processed = 0",
            (user_id,)
        )
    logger.info(f"Marked feedback as processed for user {user_id}")

def count_unprocessed_feedback(user_id: str) -> int:
    """
    Count how many unprocessed (not yet used for retraining) feedback items a user has.
    
    WHAT IT DOES:
    Counts all the movie ratings a user has given that haven't been used to
    update their profile yet. This helps determine if retraining should be triggered.
    
    HOW IT WORKS:
    1. Connects to the database
    2. Counts all feedback records where:
       - user_id matches
       - processed = 0 (not yet used)
    3. Returns the count as an integer
    
    WHEN IT'S CALLED:
    Called by check_and_retrain() to see if enough feedback has accumulated
    to trigger retraining (needs RETRAIN_THRESHOLD items).
    
    EXAMPLE:
        count_unprocessed_feedback("sarah")
        # Returns: 2
        # Meaning: Sarah has 2 ratings that haven't been used for retraining yet
        # If RETRAIN_THRESHOLD is 3, retraining won't trigger yet
    
    Args:
        user_id: String identifying which user's feedback to count
    
    Returns:
        int: Number of unprocessed feedback items
            Returns 0 if user has no unprocessed feedback
            Typical range: 0 to RETRAIN_THRESHOLD (before retraining triggers)
    
    Note:
        This is a read-only operation - it doesn't modify any data,
        just counts existing records.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM feedback WHERE user_id = ? AND processed = 0",
            (user_id,)
        )
        count = cursor.fetchone()[0]
    
    return count

# ============================================================================
# SECTION 7: RETRAINING TRIGGER LOGIC
# ============================================================================
# Purpose: Automatically retrain when enough feedback accumulates
# Why: In production, you don't manually trigger retraining - it happens automatically

def check_and_retrain(user_id: str):
    """
    Check if enough feedback has been collected, and retrain the user's profile if so.
    Uses a sliding window approach - only recent feedback is used for retraining.
    
    WHAT IT DOES:
    This is the "trigger mechanism" for the learning loop. It checks if a user
    has given enough movie ratings (RETRAIN_THRESHOLD), and if so, updates their
    profile based on those ratings. Uses a sliding window to only consider
    recent feedback (last SLIDING_WINDOW_DAYS days).
    
    HOW IT WORKS (Step by step):
    1. Counts how many unprocessed feedback items the user has
    2. Gets recent feedback within the sliding window (last N days)
    3. If unprocessed_count >= RETRAIN_THRESHOLD AND recent_feedback >= MIN_FEEDBACK_FOR_RETRAIN:
       a. Gets the user's current profile
       b. Gets all recent unprocessed feedback (within sliding window)
       c. Calls retrain_profile() to calculate new preferences
       d. Saves the updated profile to database
       e. Marks all feedback as processed (so it's not used again)
       f. Returns True (retraining happened)
    4. If conditions not met:
       - Does nothing
       - Returns False (retraining didn't happen)
    
    SLIDING WINDOW BENEFITS:
    - Only uses recent feedback (last 90 days by default)
    - Adapts to changing user preferences over time
    - Prevents old, irrelevant feedback from affecting current recommendations
    - Ensures model stays current with user's evolving tastes
    
    EXAMPLE SCENARIO:
        RETRAIN_THRESHOLD = 3
        SLIDING_WINDOW_DAYS = 90
        
        User gives 1st rating (today) → count = 1 → No retraining (1 < 3)
        User gives 2nd rating (today) → count = 2 → No retraining (2 < 3)
        User gives 3rd rating (today) → count = 3 → RETRAINING TRIGGERED! (3 >= 3)
        
        After retraining:
        - Profile updated based on all 3 recent ratings
        - All 3 ratings marked as processed
        - Only feedback from last 90 days was considered
        
        If user has 100 old ratings (120 days ago) and 2 new ratings:
        - Only the 2 new ratings are considered
        - Old ratings are ignored (outside sliding window)
        - Retraining won't trigger until 3rd recent rating
    
    WHY THIS MATTERS:
    This function automates the learning process with time-awareness. Instead of
    manually retraining after each rating, we wait until enough recent feedback
    accumulates. This:
    - Makes the system more efficient (batch processing)
    - Prevents overreacting to single ratings
    - Ensures we have enough recent data to make meaningful profile updates
    - Adapts to changing user preferences over time
    
    Args:
        user_id: String identifying which user to check and potentially retrain
    
    Returns:
        bool: True if retraining was triggered and completed
             False if not enough feedback collected yet
    
    Side Effects:
        - May update user profile in database
        - May mark feedback as processed
        - Logs retraining activity for monitoring
    """
    unprocessed_count = count_unprocessed_feedback(user_id)
    
    # Get recent unprocessed feedback within sliding window
    cutoff_date = datetime.now() - timedelta(days=SLIDING_WINDOW_DAYS)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT movie_name, rating, timestamp 
            FROM feedback 
            WHERE user_id = ? AND processed = 0 AND timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (user_id, cutoff_date.isoformat())
        )
        results = cursor.fetchall()
    
    recent_unprocessed_feedback = [
        {
            'movie_name': row[0],
            'rating': row[1],
            'days_ago': (datetime.now() - datetime.fromisoformat(row[2])).days if isinstance(row[2], str) else (datetime.now() - row[2]).days
        }
        for row in results
    ]
    
    recent_count = len(recent_unprocessed_feedback)
    
    if unprocessed_count >= RETRAIN_THRESHOLD and recent_count >= MIN_FEEDBACK_FOR_RETRAIN:
        logger.info(f"🔄 RETRAINING TRIGGERED for {user_id} ({unprocessed_count} unprocessed, {recent_count} recent within {SLIDING_WINDOW_DAYS} days)")
        
        # Get current profile
        current_profile = get_user_profile(user_id)
        
        # Use only recent unprocessed feedback for retraining
        # Remove extra fields (days_ago, timestamp) that retrain_profile doesn't need
        feedback_data = [
            {'movie_name': item['movie_name'], 'rating': item['rating']}
            for item in recent_unprocessed_feedback
        ]
        
        # Retrain the model with recent feedback only (sliding window)
        updated_profile = retrain_profile(current_profile, feedback_data)
        
        # Save updated profile
        save_user_profile(user_id, updated_profile)
        
        # Mark feedback as processed
        mark_feedback_processed(user_id)
        
        logger.info(f"✓ Retraining complete. Profile updated from {current_profile} to {updated_profile}")
        logger.info(f"  Used {len(feedback_data)} recent feedback items (last {SLIDING_WINDOW_DAYS} days)")
        return True
    else:
        if recent_count < MIN_FEEDBACK_FOR_RETRAIN:
            logger.info(f"Retraining not triggered: {recent_count} recent feedback < {MIN_FEEDBACK_FOR_RETRAIN} minimum (window: {SLIDING_WINDOW_DAYS} days)")
        else:
            logger.info(f"Retraining not triggered ({unprocessed_count}/{RETRAIN_THRESHOLD} feedback collected)")
        return False

# ============================================================================
# SECTION 8: FASTAPI APPLICATION (API Layer)
# ============================================================================
# Purpose: Expose the system as HTTP endpoints
# Why: Production systems need APIs for other services to interact with

app = FastAPI(
    title="Movie Recommendation Learning Loop API",
    description="Production-lite ML system with continuous learning",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Initialize the system when the API server starts.
    
    WHAT IT DOES:
    This function runs automatically when the FastAPI server starts up.
    It ensures the database and all tables are created before the API
    starts accepting requests.
    
    HOW IT WORKS:
    1. Calls init_database() to create tables if they don't exist
    2. Logs that the API has started successfully
    
    WHEN IT RUNS:
    Automatically called by FastAPI when you start the server with uvicorn.
    You don't call this function directly.
    
    WHY IT'S NEEDED:
    Without this, if the database doesn't exist or tables are missing,
    the API would crash when trying to handle requests. This ensures
    everything is set up correctly before the server goes "live".
    
    Returns:
        None (just initializes, doesn't return anything)
    """
    init_database()
    logger.info("🚀 API started successfully")

@app.get("/")
async def root():
    """
    Health check endpoint - verify the API is running.
    
    WHAT IT DOES:
    A simple endpoint that returns basic information about the API.
    This is useful for checking if the server is running and what
    version of the service you're connected to.
    
    HOW IT WORKS:
    Returns a simple JSON response with status information.
    No database queries or complex logic - just returns static info.
    
    EXAMPLE REQUEST:
        GET /
    
    EXAMPLE RESPONSE:
        {
            "status": "online",
            "service": "Movie Recommendation Learning Loop",
            "version": "1.0.0"
        }
    
    WHEN IT'S USEFUL:
    - Health checks: Monitoring systems can call this to verify the API is up
    - Quick verification: Developers can quickly check if server is running
    - Version checking: See what version of the API is deployed
    
    Returns:
        dict: Contains:
            - status: "online" (indicates API is running)
            - service: Name of the service
            - version: Version number of the API
    
    Note:
        This endpoint doesn't require any parameters and doesn't access
        the database. It's the simplest endpoint in the API.
    """
    return {
        "status": "online",
        "service": "Movie Recommendation Learning Loop",
        "version": "1.0.0"
    }

@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
async def get_recommendation(user_id: str):
    """
    Get a movie recommendation for a specific user.
    
    WHAT IT DOES:
    This is the main "inference" endpoint - it recommends a movie to a user based
    on their current preference profile. This is what users call when they want
    a movie suggestion.
    
    HOW IT WORKS (Step by step):
    1. Gets the user's current profile from the database
       (creates a default profile if user is new)
    2. Calculates similarity score for EVERY movie in the catalog
    3. Finds the movie with the LOWEST similarity score (best match)
    4. Returns the recommendation along with:
       - The recommended movie name
       - The similarity score (how good the match is)
       - The user's current profile
       - All similarity scores for all movies (for debugging)
    
    EXAMPLE REQUEST:
        GET /recommend/sarah
        
    EXAMPLE RESPONSE:
        {
            "user_id": "sarah",
            "recommended_movie": "Die Hard",
            "similarity_score": 3.0,
            "user_profile": [5.0, 2.0, 1.0],
            "all_scores": {
                "Die Hard": 3.0,
                "The Notebook": 8.0,
                "Blade Runner": 6.0,
                ...
            }
        }
    
    WHAT IS INFERENCE?
    "Inference" means using the learned model (user profile) to make a prediction
    (movie recommendation). It's different from "training" which updates the model.
    
    WHEN IT'S CALLED:
    - When a user opens the app and wants a recommendation
    - After a user rates movies (to see updated recommendations)
    - Can be called anytime to get current best match
    
    Args:
        user_id: String path parameter - identifies which user wants a recommendation
                Example: /recommend/sarah → user_id = "sarah"
    
    Returns:
        RecommendationResponse: A Pydantic model containing:
            - user_id: The user who got the recommendation
            - recommended_movie: Name of the best matching movie
            - similarity_score: How well it matches (lower = better)
            - user_profile: The user's current preferences [action, romance, sci-fi]
            - all_scores: Dictionary of all movies and their similarity scores
    
    Raises:
        HTTPException: 500 error if something goes wrong (database error, etc.)
    
    Note:
        This endpoint is read-only - it doesn't modify any data, just reads
        the profile and calculates recommendations.
    """
    try:
        # Get user's current profile
        profile = get_user_profile(user_id)
        
        # Calculate similarity for all movies
        scores = {}
        best_movie = None
        best_score = float('inf')
        
        for movie_name, movie_embedding in MOVIES.items():
            score = calculate_similarity(profile, movie_embedding)
            scores[movie_name] = round(score, 2)
            
            if score < best_score:
                best_score = score
                best_movie = movie_name
        
        logger.info(f"Recommendation served for {user_id}: {best_movie}")
        
        return RecommendationResponse(
            user_id=user_id,
            recommended_movie=best_movie,
            similarity_score=round(best_score, 2),
            user_profile=profile,
            all_scores=scores
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit a user's movie rating (like/dislike).
    
    WHAT IT DOES:
    This is the "feedback collection" endpoint. Users call this after watching
    a movie to rate it. The system saves the rating and may trigger retraining
    if enough feedback has been collected.
    
    HOW IT WORKS (Step by step):
    1. Validates the request:
       - Checks that the movie exists in the catalog
       - Checks that rating is 0 or 1 (not some other number)
    2. Saves the feedback to the database (marked as unprocessed)
    3. Checks if retraining should be triggered:
       - Counts unprocessed feedback
       - If count >= RETRAIN_THRESHOLD, triggers retraining
    4. Returns a response indicating:
       - Whether feedback was saved successfully
       - Whether retraining was triggered
       - How many unprocessed feedback items remain
    
    EXAMPLE REQUEST:
        POST /feedback
        Body (JSON):
        {
            "user_id": "sarah",
            "movie_name": "Die Hard",
            "rating": 1
        }
    
    EXAMPLE RESPONSE:
        {
            "status": "success",
            "message": "Feedback recorded: Die Hard - 👍",
            "retrained": false,
            "unprocessed_feedback_count": 1
        }
        
        (After 3rd rating):
        {
            "status": "success",
            "message": "Feedback recorded: The Matrix - 👍",
            "retrained": true,  // Retraining was triggered!
            "unprocessed_feedback_count": 0  // All processed now
        }
    
    THE LEARNING LOOP:
    This endpoint is part of the continuous learning loop:
    1. User gets recommendation → watches movie
    2. User calls this endpoint → rates the movie
    3. System saves rating → checks if retraining needed
    4. If enough ratings → retraining updates profile
    5. Next recommendation → uses updated (better) profile
    
    Args:
        request: FeedbackRequest object containing:
                - user_id: str - Who gave this rating
                - movie_name: str - Which movie was rated
                - rating: int - 1 = liked, 0 = disliked
    
    Returns:
        dict: Response containing:
            - status: "success" if everything worked
            - message: Human-readable confirmation message
            - retrained: bool - Whether retraining was triggered
            - unprocessed_feedback_count: int - How many ratings waiting to be processed
    
    Raises:
        HTTPException: 
            - 404 if movie not found in catalog
            - 400 if rating is not 0 or 1
            - 500 if database error occurs
    
    Side Effects:
        - Saves feedback to database
        - May trigger profile retraining
        - May mark feedback as processed
    """
    try:
        # Validate movie exists
        if request.movie_name not in MOVIES:
            raise HTTPException(status_code=404, detail=f"Movie '{request.movie_name}' not found")
        
        # Validate rating
        if request.rating not in [0, 1]:
            raise HTTPException(status_code=400, detail="Rating must be 0 (dislike) or 1 (like)")
        
        # Save feedback
        save_feedback(request.user_id, request.movie_name, request.rating)
        
        # Check if we should retrain
        retrained = check_and_retrain(request.user_id)
        
        return {
            "status": "success",
            "message": f"Feedback recorded: {request.movie_name} - {'👍' if request.rating == 1 else '👎'}",
            "retrained": retrained,
            "unprocessed_feedback_count": count_unprocessed_feedback(request.user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/profile")
async def get_profile(user_id: str):
    """
    Get a user's current preference profile and retraining status.
    
    WHAT IT DOES:
    Returns detailed information about a user's profile and how much feedback
    they've given. This is useful for debugging and monitoring the system.
    
    HOW IT WORKS:
    1. Retrieves the user's profile from the database
    2. Counts how many unprocessed feedback items they have
    3. Returns all this information in a readable format
    
    EXAMPLE REQUEST:
        GET /user/sarah/profile
    
    EXAMPLE RESPONSE:
        {
            "user_id": "sarah",
            "profile": [4.2, 2.1, 1.8],  // Current preferences
            "unprocessed_feedback": 2,     // 2 ratings waiting to be processed
            "retrain_threshold": 3         // Need 3 total to trigger retraining
        }
    
    WHEN IT'S USEFUL:
    - Debugging: See what a user's profile looks like
    - Monitoring: Check how close users are to triggering retraining
    - Development: Understand how profiles evolve over time
    
    Args:
        user_id: String path parameter - identifies which user's profile to view
    
    Returns:
        dict: Contains:
            - user_id: The user's identifier
            - profile: List of 3 floats [action, romance, sci-fi] - current preferences
            - unprocessed_feedback: int - Count of ratings not yet used for retraining
            - retrain_threshold: int - How many ratings needed to trigger retraining
    
    Raises:
        HTTPException: 500 error if database error occurs
    
    Note:
        This is a read-only endpoint - it doesn't modify any data.
    """
    try:
        profile = get_user_profile(user_id)
        unprocessed = count_unprocessed_feedback(user_id)
        
        return {
            "user_id": user_id,
            "profile": profile,
            "unprocessed_feedback": unprocessed,
            "retrain_threshold": RETRAIN_THRESHOLD
        }
    except Exception as e:
        logger.error(f"Error retrieving profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies")
async def list_movies():
    """
    Get a list of all movies available in the recommendation system.
    
    WHAT IT DOES:
    Returns information about every movie in the catalog, including their
    characteristics (embeddings) in a human-readable format.
    
    HOW IT WORKS:
    1. Iterates through all movies in the MOVIES dictionary
    2. For each movie, extracts:
       - Movie name
       - Embedding vector (the 3 numbers)
       - Human-readable characteristics (action, romance, sci-fi scores)
    3. Returns everything in a structured format
    
    EXAMPLE REQUEST:
        GET /movies
    
    EXAMPLE RESPONSE:
        {
            "total_movies": 5,
            "movies": [
                {
                    "name": "Die Hard",
                    "embedding": [5, 1, 0],
                    "characteristics": {
                        "action": 5,
                        "romance": 1,
                        "sci-fi": 0
                    }
                },
                {
                    "name": "The Notebook",
                    "embedding": [0, 5, 1],
                    "characteristics": {
                        "action": 0,
                        "romance": 5,
                        "sci-fi": 1
                    }
                },
                ...
            ]
        }
    
    WHEN IT'S USEFUL:
    - Users can see what movies are available
    - Developers can verify movie data
    - Helps understand what characteristics each movie has
    
    Returns:
        dict: Contains:
            - total_movies: int - Number of movies in catalog
            - movies: List of dicts, each containing:
                - name: str - Movie title
                - embedding: List[int] - The 3-number vector [action, romance, sci-fi]
                - characteristics: dict - Human-readable breakdown of the embedding
    
    Note:
        This is a read-only endpoint. Movies are defined in the MOVIES constant
        at the top of the file, not in the database.
    """
    return {
        "total_movies": len(MOVIES),
        "movies": [
            {
                "name": name,
                "embedding": embedding,
                "characteristics": {
                    "action": embedding[0],
                    "romance": embedding[1],
                    "sci-fi": embedding[2]
                }
            }
            for name, embedding in MOVIES.items()
        ]
    }

# ============================================================================
# SECTION 9: HOW TO RUN THIS IN PRODUCTION-LITE MODE
# ============================================================================
"""
INSTALLATION:
1. pip install fastapi uvicorn

RUNNING THE SERVER:
uvicorn movie_recommendation_lesson6:app --reload

API ENDPOINTS:
- GET  /                          - Health check
- GET  /recommend/{user_id}       - Get recommendation for a user
- POST /feedback                  - Submit user rating (triggers learning)
- GET  /user/{user_id}/profile    - View user's current profile
- GET  /movies                    - List all movies

EXAMPLE USAGE (using curl or a tool like Postman):

# Get recommendation for Sarah
curl http://localhost:8000/recommend/sarah

# Submit feedback (Sarah liked Die Hard)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "sarah", "movie_name": "Die Hard", "rating": 1}'

# Submit more feedback (will trigger retraining after 3 ratings)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "sarah", "movie_name": "The Matrix", "rating": 1}'

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "sarah", "movie_name": "The Notebook", "rating": 0}'

# Get updated recommendation (should use retrained profile)
curl http://localhost:8000/recommend/sarah

# View Sarah's profile
curl http://localhost:8000/user/sarah/profile

WHAT TO NOTICE:
- The database persists between restarts (movie_recommendations.db file)
- Retraining happens automatically after RETRAIN_THRESHOLD feedback items
- Logging shows you what's happening in real-time
- API documentation is auto-generated at http://localhost:8000/docs
- Multiple users can use the system simultaneously
- Each user has their own profile that evolves independently

THE LEARNING LOOP IN PRODUCTION:
1. User calls /recommend/{user_id} → Gets recommendation (INFERENCE)
2. User watches movie and rates it
3. User calls /feedback → Stores rating (FEEDBACK COLLECTION)
4. System checks if threshold met → Triggers retraining (LEARNING)
5. User's next call to /recommend/{user_id} → Uses updated profile (IMPROVED INFERENCE)
6. Loop continues...

This is the SAME learning loop as the simple version, but now:
- It runs continuously (API always available)
- It persists data (survives restarts)
- It scales to multiple users
- It has monitoring (logs)
- It has automatic triggers (threshold-based retraining)
"""

