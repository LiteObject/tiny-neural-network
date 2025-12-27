"""
A simple neural network implementation for learning basic patterns.

This module provides a basic neural network class that can be trained
using backpropagation to recognize patterns in binary input data.
"""

import math
import random
from typing import TypedDict


class TrainingData(TypedDict):
    """Type definition for training data entries."""

    inputs: list[int]
    output: int


class NeuralNetwork:
    """A simple single-layer neural network.

    This class implements a basic neural network with sigmoid activation
    that can learn to classify binary input patterns.

    Attributes:
        weights: The synaptic weights of the neural network.
    """

    def __init__(self) -> None:
        """Initialize the neural network with random weights."""

        random.seed(1)

        self.weights: list[float] = [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1),
        ]

    def think(self, neuron_inputs: list[int]) -> float:
        """Process inputs through the neural network to produce an output.

        Args:
            neuron_inputs: A list of input values to process.

        Returns:
            The neural network's prediction (0 to 1) based on current weights.
        """
        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_inputs)
        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        return neuron_output

    def train(
        self, training_set_data: list[TrainingData], number_of_iterations: int
    ) -> None:
        """Train the neural network using backpropagation.

        Adjusts the weights based on the error between predicted
        and expected outputs.

        Args:
            training_set_data: A list of training examples, each containing
                inputs and expected output.
            number_of_iterations: The number of training iterations to perform.
        """
        for _ in range(number_of_iterations):
            # Process each training example in the dataset
            for training_set in training_set_data:

                # Forward pass: get the network's prediction for this input
                predicted_output = self.think(training_set["inputs"])

                # Calculate how wrong the prediction was (positive = too low, negative = too high)
                error_in_output = training_set["output"] - predicted_output

                # Update each weight based on its contribution to the error
                for index, _ in enumerate(self.weights):

                    # Get the input value associated with this weight
                    neuron_input = training_set["inputs"][index]

                    # Calculate weight adjustment using gradient descent:
                    # - neuron_input: how much this input contributed to the output
                    # - error_in_output: how much we need to correct
                    # - sigmoid_gradient: how confident the network was (less adjustment when confident)
                    adjust_weight = (
                        neuron_input
                        * error_in_output
                        * self.__sigmoid_gradient(predicted_output)
                    )

                    # Apply the adjustment to move the weight in the right direction
                    self.weights[index] += adjust_weight

    def __sigmoid(self, sum_of_weighted_inputs: float) -> float:
        """Apply the sigmoid activation function.

        The sigmoid function maps any value to a value between 0 and 1,
        creating an S-shaped curve.

        Args:
            sum_of_weighted_inputs: The weighted sum to transform.

        Returns:
            A float between 0 and 1.
        """
        if sum_of_weighted_inputs < -700:  # Prevent overflow
            return 0.0

        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

    def __sigmoid_gradient(self, neuron_output: float) -> float:
        """Calculate the gradient of the sigmoid function.

        Used during backpropagation to determine how much to adjust weights.

        Args:
            neuron_output: The output of the sigmoid function.

        Returns:
            The derivative of the sigmoid function at the given point.
        """
        return neuron_output * (1 - neuron_output)

    def __sum_of_weighted_inputs(self, neuron_inputs: list[int]) -> float:
        """Calculate the weighted sum of inputs.

        Computes the dot product of inputs and weights.

        Args:
            neuron_inputs: A list of input values.

        Returns:
            The sum of each input multiplied by its corresponding weight.
        """
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs


TRAINING_SET_DATA: list[TrainingData] = [
    {"inputs": [0, 0, 1], "output": 0},
    {"inputs": [1, 1, 1], "output": 1},
    {"inputs": [1, 0, 1], "output": 1},
    {"inputs": [0, 1, 1], "output": 0},
]

if __name__ == "__main__":
    # Create a neural network
    nn = NeuralNetwork()

    print("Initial weights:")
    print(nn.weights)

    # Train the network
    print("\nTraining the neural network...")
    nn.train(TRAINING_SET_DATA, number_of_iterations=10000)

    print("\nWeights after training:")
    print(nn.weights)

    # Test the network
    print("\nTesting the neural network:")
    for data in TRAINING_SET_DATA:
        result = nn.think(data["inputs"])
        print(
            f"Input: {data['inputs']} => Prediction: {result:.6f} (Expected: {data['output']})"
        )

    # Test with a new input
    new_input = [1, 1, 0]
    prediction = nn.think(new_input)
    print(f"\nNew input: {new_input} => Prediction: {prediction:.6f}")
