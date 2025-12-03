# Movie Recommendation Learning Loop

A production-lite movie recommendation system that demonstrates continuous learning from user feedback.

## Overview

This project implements a machine learning recommendation system that:
- Uses embedding vectors to represent user preferences and movie characteristics
- Recommends movies based on similarity calculations
- Learns from user feedback (likes/dislikes) to improve recommendations
- Automatically retrains user profiles when enough feedback is collected

## Features

- **FastAPI REST API** - Production-ready HTTP endpoints
- **SQLite Database** - Persistent storage for user profiles and feedback
- **Automatic Retraining** - Triggers learning after collecting N feedback items
- **Comprehensive Documentation** - Novice-friendly explanations of all functions
- **Type Hints** - Full type annotations for better code quality
- **Context Managers** - Safe database connection handling

## Installation

```bash
pip install fastapi uvicorn
```

## Running the Server

```bash
uvicorn movie_recommendation_lesson6:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `GET /recommend/{user_id}` - Get movie recommendation for a user
- `POST /feedback` - Submit user rating (triggers learning)
- `GET /user/{user_id}/profile` - View user's current profile
- `GET /movies` - List all available movies

## API Documentation

Once the server is running, visit:
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

## Example Usage

### Get a recommendation
```bash
curl http://localhost:8000/recommend/sarah
```

### Submit feedback
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"user_id": "sarah", "movie_name": "Die Hard", "rating": 1}'
```

## How It Works

### The Learning Loop

1. **Inference**: User requests a recommendation → System calculates similarity scores
2. **Feedback Collection**: User rates the movie → System stores the rating
3. **Learning**: After N ratings → System updates user profile based on feedback
4. **Improved Inference**: Next recommendation uses the updated profile

### Similarity Calculation

Uses Manhattan distance to compare user preferences with movie characteristics:
- Lower score = better match
- Formula: Sum of absolute differences across all dimensions

### Profile Retraining

- **Liked movies** (rating=1): Pull user profile toward movie's characteristics
- **Disliked movies** (rating=0): Push user profile away from movie's characteristics
- **Learning rate**: 0.3 (30% adjustment per feedback)

## Code Structure

- **Database Layer**: Safe connection handling with context managers
- **Business Logic**: Similarity calculation and profile retraining
- **API Layer**: FastAPI endpoints for inference and feedback
- **Data Models**: Pydantic schemas for request/response validation

## Documentation

Every function includes comprehensive documentation with:
- What it does
- How it works (step-by-step)
- Examples with concrete numbers
- Parameter descriptions
- Return value explanations

Perfect for learning and understanding the codebase!

## License

MIT

