"""
Simple Rating Predictor - Linear Regression Learning Module

WHAT THIS MODULE TEACHES:
This module demonstrates how to build a simple machine learning model that learns
to predict movie ratings based on movie features. It's a hands-on introduction to
linear regression - one of the most fundamental machine learning techniques.

WHAT IS LINEAR REGRESSION?
Linear regression is a way to find a mathematical formula (a "model") that can
predict a number (like a movie rating) based on other numbers (like how much action,
romance, or comedy a movie has). Think of it like finding the best recipe:

    Rating = (weight1 × action_level) + (weight2 × romance_level) + (weight3 × comedy_level) + bias

The model "learns" by adjusting the weights and bias until its predictions match
the actual ratings we've seen. This is called "training" the model.

HOW IT WORKS:
1. We start with random weights (the model doesn't know anything yet)
2. We show the model many examples of movies and their ratings
3. For each example, the model makes a prediction
4. We calculate how wrong the prediction was (the "error")
5. We adjust the weights slightly to make better predictions next time
6. We repeat this many times until the model gets good at predicting

WHY LEARN THIS?
- Linear regression is the foundation of many advanced ML techniques
- It teaches you the core concepts: features, weights, predictions, training
- It's simple enough to understand completely, yet powerful enough to be useful
- The same ideas (gradient descent, loss functions) appear in neural networks

WHAT YOU'LL LEARN:
- How to make predictions from features and weights
- How to train a model using gradient descent
- How to measure how well a model is learning (loss)
- How to visualize training progress

PREREQUISITES:
- Basic Python knowledge (functions, classes, loops)
- Understanding of lists and arrays
- Familiarity with basic math (multiplication, addition)

RUNNING THIS MODULE:
    python simple_rating_predictor.py

This will:
1. Generate synthetic training data (movies with known rating patterns)
2. Create a model and train it
3. Show how the model learns the correct weights
4. Display a plot showing training progress
"""

import numpy as np
import matplotlib.pyplot as plt


class SimpleRatingPredictor:
    """
    A basic machine learning model that learns to predict movie ratings.
    
    WHAT IT DOES:
    This class represents a simple linear regression model. It learns to predict
    how much someone will like a movie (a rating) based on the movie's features
    (like how much action, romance, or comedy it has).
    
    HOW IT WORKS (The Big Picture):
    The model stores "weights" - one for each feature. When you give it a movie's
    features, it multiplies each feature by its corresponding weight, adds them
    all up, and adds a "bias" term. This gives you the predicted rating.
    
    EXAMPLE - Before Training:
        Movie features: [action=0.8, romance=0.2, comedy=0.1]
        Weights (random): [0.01, -0.02, 0.005]
        Bias: 0.0
        
        Prediction = (0.8 × 0.01) + (0.2 × -0.02) + (0.1 × 0.005) + 0.0
                   = 0.008 - 0.004 + 0.0005
                   = 0.0045  (not very good - weights are random!)
    
    EXAMPLE - After Training:
        Movie features: [action=0.8, romance=0.2, comedy=0.1]
        Weights (learned): [2.0, 1.0, -0.5]
        Bias: 3.0
        
        Prediction = (0.8 × 2.0) + (0.2 × 1.0) + (0.1 × -0.5) + 3.0
                   = 1.6 + 0.2 - 0.05 + 3.0
                   = 4.75  (much better! close to true rating)
    
    WHY WE USE THIS APPROACH:
    - Simple and interpretable: you can see exactly what the model is doing
    - Fast to train: doesn't require powerful computers
    - Foundation for more complex models: neural networks build on these ideas
    - Works well when relationships are roughly linear (straight-line patterns)
    
    Attributes:
        weights (np.ndarray): Array of weights, one for each feature.
                             These are learned during training.
                             Shape: (n_features,)
        bias (float): A constant term added to every prediction.
                     Also learned during training.
                     Think of it as a "baseline rating".
    """
    
    def __init__(self, n_features):
        """
        Initialize a new rating predictor model.
        
        WHAT IT DOES:
        Creates a new model with random starting weights and zero bias. The model
        doesn't know anything useful yet - it will learn from training data.
        
        HOW IT WORKS:
        1. Creates an array of random weights (one for each feature)
        2. Makes the weights small (multiplied by 0.01) so predictions start near zero
        3. Sets bias to 0.0 initially
        
        WHY SMALL RANDOM WEIGHTS?
        - Starting with zeros would mean the model can't learn (all gradients would be zero)
        - Starting with large random values would make training unstable
        - Small random values give the model a "gentle push" in random directions
        - The model will quickly adjust these during training
        
        EXAMPLE:
            model = SimpleRatingPredictor(n_features=3)
            # Creates model with 3 weights (for action, romance, comedy)
            # Weights might be: [0.012, -0.008, 0.005]
            # Bias: 0.0
        
        Args:
            n_features (int): Number of features each movie has.
                            For example, if features are [action, romance, comedy],
                            then n_features = 3.
                            Must be a positive integer.
        
        Returns:
            None (this is a constructor - it modifies self, doesn't return a value)
        """
        # Initialize weights randomly - these will be learned
        # np.random.randn() generates random numbers from a normal distribution
        # Multiplying by 0.01 makes them small (between roughly -0.03 and +0.03)
        self.weights = np.random.randn(n_features) * 0.01
        
        # Start with zero bias - will be learned during training
        self.bias = 0.0
    
    def predict(self, features):
        """
        Make a prediction: estimate what rating this movie would get.
        
        WHAT IT DOES:
        Takes a movie's features (like [action_level, romance_level, comedy_level])
        and predicts what rating it would receive. This is the "forward pass" -
        using the current weights to make a prediction.
        
        HOW IT WORKS (The Math):
        The prediction is a weighted sum of features plus a bias:
        
            prediction = (feature1 × weight1) + (feature2 × weight2) + ... + bias
            prediction = sum of (each feature × its weight) + bias
        
        In mathematical notation:
            prediction = features[0] × weights[0] + features[1] × weights[1] + ... + bias
        
        Or using vector notation (what we'll actually do):
            prediction = features @ weights + bias
            (where @ means "dot product" - multiply and sum)
        
        STEP-BY-STEP EXAMPLE:
        Let's say we have:
            features = [0.8, 0.2, 0.1]  (high action, low romance, low comedy)
            weights = [2.0, 1.0, -0.5]
            bias = 3.0
        
        Step 1: Multiply each feature by its weight
            action_contribution = 0.8 × 2.0 = 1.6
            romance_contribution = 0.2 × 1.0 = 0.2
            comedy_contribution = 0.1 × -0.5 = -0.05
        
        Step 2: Add all contributions together
            sum = 1.6 + 0.2 + (-0.05) = 1.75
        
        Step 3: Add the bias
            prediction = 1.75 + 3.0 = 4.75
        
        So the model predicts a rating of 4.75 for this movie!
        
        YOUR TASK:
        Implement this calculation. You can use:
        - NumPy's dot product: np.dot(features, self.weights)
        - Or the @ operator: features @ self.weights
        - Then add self.bias
        
        HINT:
        If features is [0.8, 0.2, 0.1] and weights is [2.0, 1.0, -0.5]:
            result = 0.8*2.0 + 0.2*1.0 + 0.1*(-0.5) + bias
            result = 1.6 + 0.2 - 0.05 + bias
        
        Args:
            features (np.ndarray): Array of feature values for one movie.
                                 Shape: (n_features,)
                                 Example: [0.8, 0.2, 0.1] for [action, romance, comedy]
        
        Returns:
            float: The predicted rating for this movie.
                 Could be any number (positive or negative, though ratings are usually positive).
                 After training, should be close to actual ratings.
        
        Note:
            This method doesn't modify the model - it just uses current weights to predict.
            To improve predictions, you need to call train_step() many times.
        """
        # IMPLEMENTATION WITH HINTS:
        # Calculate prediction: prediction = sum(features * weights) + bias
        
        # Method 1 (element-wise, then sum) - easier to understand:
        # Step 1: Multiply each feature by its corresponding weight
        #   products = features * self.weights  # [feature[0]*weight[0], feature[1]*weight[1], ...]
        # Step 2: Sum all the products
        #   sum_products = np.sum(products)
        # Step 3: Add the bias
        #   prediction = sum_products + self.bias
        
        # Method 2 (dot product - recommended, more efficient):
        # The dot product does multiplication and summation in one step!
        # np.dot(features, self.weights) calculates: features[0]*weights[0] + features[1]*weights[1] + ...
        prediction = np.dot(features, self.weights) + self.bias
        # Alternative syntax (same thing): prediction = features @ self.weights + self.bias
        
        return prediction
    
    def train_step(self, features, true_rating, learning_rate=0.01):
        """
        One training step: predict, calculate error, update weights.
        
        WHAT IT DOES:
        This is the "learning" part! The model makes a prediction, sees how wrong
        it was, and adjusts its weights to do better next time. This is called
        "gradient descent" - we're descending (going down) the "error hill" to
        find the best weights.
        
        HOW IT WORKS (The Training Process):
        
        Step 1: Make a prediction using current weights
            prediction = self.predict(features)
        
        Step 2: Calculate how wrong we were (the error)
            error = prediction - true_rating
            (If prediction was too high, error is positive. If too low, error is negative)
        
        Step 3: Calculate how much to adjust each weight
            For each weight i:
                adjustment = learning_rate × error × features[i]
                new_weight[i] = old_weight[i] - adjustment
        
        Step 4: Adjust the bias similarly
            bias_adjustment = learning_rate × error
            new_bias = old_bias - bias_adjustment
        
        Step 5: Calculate the loss (squared error) to track progress
            loss = error² = (prediction - true_rating)²
        
        WHY THIS WORKS (The Intuition):
        - If our prediction was too high (error > 0), we reduce the weights
        - If our prediction was too low (error < 0), we increase the weights
        - The learning_rate controls how big our steps are (too big = unstable, too small = slow)
        - We multiply by the feature value because features that are larger should
          have more influence on the adjustment
        
        EXAMPLE - One Training Step:
        Let's say:
            features = [0.8, 0.2, 0.1]
            true_rating = 5.0
            current weights = [1.0, 1.0, 1.0]
            current bias = 0.0
            learning_rate = 0.01
        
        Step 1: Predict
            prediction = 0.8×1.0 + 0.2×1.0 + 0.1×1.0 + 0.0 = 1.1
        
        Step 2: Calculate error
            error = 1.1 - 5.0 = -3.9  (we predicted too low!)
        
        Step 3: Update weights
            weight[0] adjustment = 0.01 × (-3.9) × 0.8 = -0.0312
            new weight[0] = 1.0 - (-0.0312) = 1.0312  (increased!)
            
            weight[1] adjustment = 0.01 × (-3.9) × 0.2 = -0.0078
            new weight[1] = 1.0 - (-0.0078) = 1.0078  (increased!)
            
            weight[2] adjustment = 0.01 × (-3.9) × 0.1 = -0.0039
            new weight[2] = 1.0 - (-0.0039) = 1.0039  (increased!)
        
        Step 4: Update bias
            bias adjustment = 0.01 × (-3.9) = -0.039
            new bias = 0.0 - (-0.039) = 0.039  (increased!)
        
        Step 5: Calculate loss
            loss = (-3.9)² = 15.21
        
        After many such steps, the weights will converge to the correct values!
        
        YOUR TASK:
        Implement the training step. Follow these steps:
        
        1. Get prediction:
           prediction = self.predict(features)
        
        2. Calculate error:
           error = prediction - true_rating
        
        3. Update each weight:
           for i in range(len(self.weights)):
               adjustment = learning_rate * error * features[i]
               self.weights[i] = self.weights[i] - adjustment
           
           OR use vectorized operations:
           self.weights = self.weights - learning_rate * error * features
        
        4. Update bias:
           self.bias = self.bias - learning_rate * error
        
        5. Calculate and return squared error (loss):
           loss = error ** 2
           return loss
        
        HINT - Vectorized Update (Recommended):
        Instead of looping, you can update all weights at once:
            self.weights -= learning_rate * error * features
            self.bias -= learning_rate * error
        
        Args:
            features (np.ndarray): Feature values for one movie.
                                 Shape: (n_features,)
            true_rating (float): The actual rating this movie received.
                               This is what we're trying to predict.
            learning_rate (float): How big of steps to take when adjusting weights.
                                 Default: 0.01
                                 Smaller = slower but more stable learning
                                 Larger = faster but might overshoot
        
        Returns:
            float: The squared error (loss) for this training example.
                 This measures how wrong the prediction was.
                 Lower is better. We'll track this over time to see if training is working.
        
        Note:
            This method MODIFIES self.weights and self.bias.
            Each call makes the model slightly better (hopefully!).
            You need to call this many times (once per training example, for many epochs)
            to train the model properly.
        """
        # IMPLEMENTATION WITH HINTS:
        # This is the core learning algorithm - gradient descent!
        
        # Step 1: Get prediction using current weights
        # This calls the predict() method we just implemented
        prediction = self.predict(features)
        
        # Step 2: Calculate error (how wrong was our prediction?)
        # If prediction > true_rating: error is positive (we predicted too high)
        # If prediction < true_rating: error is negative (we predicted too low)
        error = prediction - true_rating
        
        # Step 3: Update weights using gradient descent
        # Formula: new_weight = old_weight - learning_rate * error * feature
        # 
        # Why multiply by error? If error is positive (predicted too high), we want to decrease weights
        # Why multiply by feature? Features with larger values should have bigger adjustments
        # Why multiply by learning_rate? Controls step size (too big = unstable, too small = slow)
        #
        # Vectorized version (updates all weights at once - recommended):
        self.weights = self.weights - learning_rate * error * features
        #
        # Alternative (loop version - easier to understand but slower):
        # for i in range(len(self.weights)):
        #     adjustment = learning_rate * error * features[i]
        #     self.weights[i] = self.weights[i] - adjustment
        
        # Step 4: Update bias similarly
        # Bias doesn't depend on features, so we just use: bias - learning_rate * error
        self.bias = self.bias - learning_rate * error
        
        # Step 5: Calculate and return squared error (loss)
        # We square the error to:
        # - Make it always positive (easier to work with)
        # - Penalize large errors more (2x error = 4x loss)
        loss = error ** 2
        return loss
        
        # EXAMPLE WALKTHROUGH (for understanding):
        #    features = [0.8, 0.2, 0.1]
        #    true_rating = 5.0
        #    learning_rate = 0.01
        #    current weights = [1.0, 1.0, 1.0]
        #    current bias = 0.0
        #
        #    Step 1: prediction = 0.8*1.0 + 0.2*1.0 + 0.1*1.0 + 0.0 = 1.1
        #    Step 2: error = 1.1 - 5.0 = -3.9  (we predicted too low!)
        #    Step 3: weights -= 0.01 * (-3.9) * [0.8, 0.2, 0.1]
        #            weights -= [-0.0312, -0.0078, -0.0039]
        #            weights = [1.0, 1.0, 1.0] - [-0.0312, -0.0078, -0.0039]
        #            weights = [1.0312, 1.0078, 1.0039]  (increased - good!)
        #    Step 4: bias -= 0.01 * (-3.9) = -(-0.039) = +0.039
        #            bias = 0.0 + 0.039 = 0.039  (increased - good!)
        #    Step 5: loss = (-3.9)² = 15.21
        #
        #    After many such steps, weights will converge to [2.0, 1.0, -0.5]!


def generate_training_data(n_samples=100):
    """
    Generate synthetic movie rating data for training.
    
    WHAT IT DOES:
    Creates fake (synthetic) movie data with known patterns. This is useful for
    learning because we know the "true" relationship between features and ratings,
    so we can verify that our model learns it correctly.
    
    HOW IT WORKS:
    We define a "true" formula that generates ratings:
        rating = 2.0 × action + 1.0 × romance - 0.5 × comedy + 3.0
    
    Then we:
    1. Generate random feature values (between 0 and 1)
    2. Calculate the "true" rating using the formula above
    3. Add a tiny bit of random noise (to make it realistic)
    4. Return both features and ratings
    
    WHY USE SYNTHETIC DATA?
    - We know the true pattern, so we can verify the model learns it
    - It's clean and predictable (good for learning)
    - No need to collect real data
    - We can control the difficulty (add more or less noise)
    
    EXAMPLE OUTPUT:
        features = [[0.8, 0.2, 0.1],    # Movie 1: high action, low romance/comedy
                    [0.1, 0.9, 0.3],    # Movie 2: low action, high romance
                    ...]
        ratings = [4.75, 4.15, ...]     # Corresponding true ratings
    
    THE TRUE PATTERN:
    The true weights are [2.0, 1.0, -0.5] and bias is 3.0.
    This means:
    - Action movies get +2.0 points per unit of action
    - Romance movies get +1.0 points per unit of romance
    - Comedy movies get -0.5 points per unit of comedy (people don't like comedy here!)
    - Every movie starts with a baseline of 3.0
    
    After training, our model should learn weights close to [2.0, 1.0, -0.5]
    and bias close to 3.0!
    
    Args:
        n_samples (int): How many movie examples to generate.
                        Default: 100
                        More samples = better training, but slower
    
    Returns:
        tuple: (features, ratings)
            - features (np.ndarray): Array of feature vectors.
                                    Shape: (n_samples, 3)
                                    Each row is one movie: [action, romance, comedy]
            - ratings (np.ndarray): Array of true ratings.
                                  Shape: (n_samples,)
                                  Each value is the rating for the corresponding movie
    
    Example:
        >>> features, ratings = generate_training_data(n_samples=5)
        >>> print(features.shape)  # (5, 3)
        >>> print(ratings.shape)   # (5,)
        >>> print(features[0])     # [0.8, 0.2, 0.1] (example)
        >>> print(ratings[0])      # 4.75 (approximately)
    """
    # Features: [action_level, romance_level, comedy_level]
    # Each feature is a random number between 0 and 1
    # Shape: (n_samples, 3) - n_samples rows, 3 columns
    features = np.random.rand(n_samples, 3)
    
    # True pattern: rating = 2*action + 1*romance - 0.5*comedy + 3
    # These are the "true" weights and bias that we want the model to learn
    true_weights = np.array([2.0, 1.0, -0.5])
    true_bias = 3.0
    
    # Calculate ratings using the true formula
    # @ is matrix multiplication: features @ true_weights does:
    #   for each movie: sum(features[i] * true_weights) + true_bias
    ratings = features @ true_weights + true_bias
    
    # Add a tiny bit of random noise to make it realistic
    # Real data always has some noise (measurement error, randomness, etc.)
    # std=0.1 means most noise is between -0.2 and +0.2
    ratings += np.random.randn(n_samples) * 0.1
    
    return features, ratings


def train_model(model, features, ratings, epochs=100):
    """
    Train the model and track progress over time.
    
    WHAT IT DOES:
    This function runs the training loop. It repeatedly shows the model all the
    training examples, letting it learn from its mistakes. It also tracks the
    "loss" (how wrong the predictions are) so we can see if training is working.
    
    HOW IT WORKS (The Training Loop):
    
    For each epoch (one full pass through all data):
        1. Reset the epoch loss counter
        2. For each training example (movie):
           a. Call model.train_step() to learn from this example
           b. Add the loss to our counter
        3. Calculate average loss for this epoch
        4. Print progress every 10 epochs
        5. Save the loss to track progress
    
    After all epochs:
        - The model's weights should be close to the true weights
        - The loss should have decreased (model got better)
        - We return the loss history for plotting
    
    WHY TRACK LOSS?
    - Loss tells us if training is working (should decrease over time)
    - If loss isn't decreasing, something is wrong (learning rate too high/low, bug, etc.)
    - We can plot loss over time to visualize learning progress
    - Loss helps us decide when to stop training (when it stops decreasing)
    
    EXAMPLE OUTPUT:
        Epoch 0: Loss = 15.2341
        Epoch 10: Loss = 8.1234
        Epoch 20: Loss = 3.4567
        Epoch 30: Loss = 1.2345
        ...
        Epoch 100: Loss = 0.0123  (much better!)
    
    WHAT TO EXPECT:
    - Loss should start high (model is bad at first)
    - Loss should decrease over time (model is learning)
    - Loss should eventually level off (model has learned as much as it can)
    - Final loss should be close to the noise level (0.1² = 0.01)
    
    Args:
        model (SimpleRatingPredictor): The model to train.
                                     Will be modified in place (weights updated).
        features (np.ndarray): Training features.
                             Shape: (n_samples, n_features)
                             Each row is one movie's features.
        ratings (np.ndarray): True ratings for training.
                            Shape: (n_samples,)
                            Each value is the true rating for corresponding movie.
        epochs (int): How many times to go through all training data.
                     Default: 100
                     More epochs = more learning, but takes longer
                     Too many epochs might "overfit" (memorize the data)
    
    Returns:
        list: History of average loss values, one per epoch.
            Length: epochs
            Each value is the average squared error for that epoch.
            Should decrease over time if training is working.
    
    Example:
        >>> model = SimpleRatingPredictor(n_features=3)
        >>> features, ratings = generate_training_data(n_samples=100)
        >>> losses = train_model(model, features, ratings, epochs=50)
        >>> print(f"Final loss: {losses[-1]:.4f}")  # Should be small!
        >>> print(f"Learned weights: {model.weights}")  # Should be close to [2.0, 1.0, -0.5]
    """
    losses = []  # Track loss over time to see if we're learning
    
    # An "epoch" is one full pass through all training data
    for epoch in range(epochs):
        epoch_loss = 0  # Sum of all losses in this epoch
        
        # Go through each training example (movie)
        for feature, rating in zip(features, ratings):
            # Train on this example and get the loss
            loss = model.train_step(feature, rating)
            epoch_loss += loss  # Add to our running total
        
        # Calculate average loss for this epoch
        # (average = total / number of examples)
        avg_loss = epoch_loss / len(features)
        losses.append(avg_loss)  # Save for plotting later
        
        # Print progress every 10 epochs so we can see what's happening
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    return losses


if __name__ == "__main__":
    """
    Main execution block - runs when you execute this file directly.
    
    WHAT HAPPENS WHEN YOU RUN THIS:
    1. Generates synthetic training data (movies and ratings)
    2. Creates a new model with random weights
    3. Shows initial weights (should be random/small)
    4. Trains the model for 100 epochs
    5. Shows final weights (should be close to [2.0, 1.0, -0.5])
    6. Creates a plot showing how loss decreased over time
    7. Saves the plot to 'training_progress.png'
    
    EXPECTED OUTPUT:
    You should see:
    - Initial weights: small random numbers (e.g., [0.012, -0.008, 0.005])
    - Initial bias: 0.0
    - Training progress: Loss decreasing over epochs
    - Final weights: close to [2.0, 1.0, -0.5] (the true pattern!)
    - Final bias: close to 3.0 (the true bias!)
    - A plot file showing the loss curve
    
    INTERPRETING RESULTS:
    - If final weights are close to [2.0, 1.0, -0.5]: SUCCESS! Model learned correctly
    - If loss decreases over time: Training is working!
    - If loss is very small at the end (< 0.1): Model is making good predictions
    - If weights are still random: Check your predict() and train_step() implementations!
    
    TROUBLESHOOTING:
    - If loss doesn't decrease: Check that train_step() is actually updating weights
    - If weights don't converge: Try adjusting learning_rate (might be too high or too low)
    - If you get errors: Make sure numpy and matplotlib are installed
    """
    print("=" * 60)
    print("Simple Rating Predictor - Training Demo")
    print("=" * 60)
    print()
    
    # Generate data
    print("Generating training data...")
    features, ratings = generate_training_data()
    print(f"Generated {len(features)} movie examples")
    print(f"Features shape: {features.shape} (movies × features)")
    print(f"Ratings shape: {ratings.shape} (one rating per movie)")
    print()
    
    # Create and train model
    print("Creating model...")
    model = SimpleRatingPredictor(n_features=3)
    print("Initial weights:", model.weights)
    print("Initial bias:", model.bias)
    print()
    print("Note: Weights start random - model doesn't know anything yet!")
    print()
    
    # Train the model
    print("Starting training...")
    print("(Watch the loss decrease - that means the model is learning!)")
    print()
    losses = train_model(model, features, ratings, epochs=100)
    print()
    
    # Show results
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print()
    print("Final weights:", model.weights)
    print("Final bias:", model.bias)
    print()
    print("True weights should be close to: [2.0, 1.0, -0.5]")
    print("True bias should be close to: 3.0")
    print()
    
    # Check how close we got
    true_weights = np.array([2.0, 1.0, -0.5])
    true_bias = 3.0
    
    weight_error = np.abs(model.weights - true_weights)
    bias_error = abs(model.bias - true_bias)
    
    print("How close did we get?")
    print(f"  Weight errors: {weight_error}")
    print(f"  Bias error: {bias_error:.4f}")
    print()
    
    if np.all(weight_error < 0.1) and bias_error < 0.1:
        print("✓ SUCCESS! Model learned the pattern correctly!")
    else:
        print("⚠ Model is close but not perfect. This is normal - try:")
        print("  - Training for more epochs")
        print("  - Using more training data")
        print("  - Adjusting learning rate")
    print()
    
    # Plot training progress
    print("Creating training progress plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Progress: Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotation showing final loss
    final_loss = losses[-1]
    plt.text(0.7, 0.9, f'Final Loss: {final_loss:.4f}', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("Saved training progress plot to 'training_progress.png'")
    print()
    print("=" * 60)
    print("Done! Open 'training_progress.png' to see the learning curve.")
    print("=" * 60)

