"""
A scikit-learn implementation of the perceptron for comparison.

This module demonstrates the same neural network functionality
using scikit-learn's SGDClassifier with logistic loss (sigmoid).
"""

import numpy as np
from sklearn.linear_model import SGDClassifier


# Training data (same as original)
X_TRAIN = np.array(
    [
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
    ]
)

Y_TRAIN = np.array([0, 1, 1, 0])


def create_perceptron() -> SGDClassifier:
    """
    Create a perceptron using scikit-learn's SGDClassifier.

    Uses log_loss (logistic/sigmoid) to match the original implementation.

    Returns:
        A configured SGDClassifier instance.
    """
    return SGDClassifier(
        loss="log_loss",  # Sigmoid activation (logistic regression)
        max_iter=10000,  # Same as original training iterations
        random_state=1,  # Same seed for reproducibility
        learning_rate="constant",
        eta0=1.0,  # Learning rate
    )


if __name__ == "__main__":
    # Create and train the perceptron
    nn = create_perceptron()

    print("Training the perceptron with scikit-learn...")
    nn.fit(X_TRAIN, Y_TRAIN)

    print("\nWeights after training:")
    print(f"Weights: {nn.coef_[0]}")
    print(f"Bias: {nn.intercept_[0]}")

    # Test on training data
    print("\nTesting on training data:")
    for inputs, expected in zip(X_TRAIN, Y_TRAIN):
        sample = inputs.reshape(1, -1)
        prediction = nn.predict(sample)[0]
        probability = nn.predict_proba(sample)[0][1]
        print(
            f"Input: {inputs} => Prediction: {prediction} "
            f"(Probability: {probability:.6f}, Expected: {expected})"
        )

    # Test with a new input
    new_input = np.array([[1, 0, 0]])
    prediction = nn.predict(new_input)[0]
    probability = nn.predict_proba(new_input)[0][1]
    print(
        f"\nNew input: {new_input[0]} => Prediction: {prediction} "
        f"(Probability: {probability:.6f})"
    )
