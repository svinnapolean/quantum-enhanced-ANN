"""
Classical Artificial Neural Network (ANN)

A simple but effective feedforward neural network that can learn
from quantum-enhanced training data.
"""

import numpy as np
from typing import List, Tuple, Optional


class NeuralNetwork:
    """
    A simple feedforward neural network with backpropagation.
    
    This classical ANN architecture can learn from quantum-enhanced data,
    demonstrating how quantum computing principles can enhance classical ML.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        activation: str = 'sigmoid',
        seed: Optional[int] = None
    ):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('sigmoid', 'tanh', 'relu')
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_name = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for better gradient flow
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # For storing activations during forward pass
        self.activations = []
        self.z_values = []
    
    def _activate(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        elif self.activation_name == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def _activate_derivative(self, z: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation_name == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_name == 'relu':
            return (z > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Output predictions of shape (n_samples, n_outputs)
        """
        self.activations = [X]
        self.z_values = []
        
        a = X
        for i in range(len(self.weights)):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Use sigmoid for output layer (binary classification)
            # Use chosen activation for hidden layers
            if i == len(self.weights) - 1:
                a = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # sigmoid
            else:
                a = self._activate(z)
            
            self.activations.append(a)
        
        return a
    
    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Backward propagation to compute gradients and update weights.
        
        Args:
            X: Input data
            y: True labels
        """
        m = X.shape[0]
        
        # Compute output layer error
        # For binary classification with sigmoid output
        delta = self.activations[-1] - y.reshape(-1, 1)
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights and biases
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activate_derivative(self.z_values[i - 1])
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> dict:
        """
        Train the neural network.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent (None = full batch)
            verbose: Print training progress
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        n_samples = X.shape[0]
        if batch_size is None:
            batch_size = n_samples
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch)
            
            # Compute metrics
            train_pred = self.predict(X)
            train_loss = self._compute_loss(X, y)
            train_acc = np.mean(train_pred == y)
            
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self._compute_loss(X_val, y_val)
                val_acc = np.mean(val_pred == y_val)
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        return history
    
    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute binary cross-entropy loss.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            Loss value
        """
        predictions = self.forward(X)
        y = y.reshape(-1, 1)
        
        # Binary cross-entropy
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        return loss
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input data
            
        Returns:
            Predicted class labels
        """
        predictions = self.forward(X)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input data
            
        Returns:
            Prediction probabilities
        """
        return self.forward(X).flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test data
            y: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X)
        loss = self._compute_loss(X, y)
        accuracy = np.mean(predictions == y)
        
        # Compute additional metrics
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        tn = np.sum((predictions == 0) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': [[tn, fp], [fn, tp]]
        }
