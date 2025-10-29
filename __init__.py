"""
Quantum-Enhanced Artificial Neural Network

A demonstration of how quantum mechanics simulation can enhance
classical machine learning through quantum-generated training data.
"""

from .quantum_simulator import (
    QuantumState,
    QuantumSimulator,
    QuantumDataGenerator
)

from .neural_network import NeuralNetwork

__version__ = "1.0.0"
__all__ = [
    'QuantumState',
    'QuantumSimulator', 
    'QuantumDataGenerator',
    'NeuralNetwork'
]
