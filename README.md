# Quantum-Enhanced Artificial Neural Network

This implementation demonstrates how to enhance classical Artificial Neural Networks (ANNs) with **quantum mechanics simulation** to create training data that incorporates quantum effects like **superposition**, **entanglement**, **interference**, and **measurement**. 

## Overview

This project bridges quantum computing principles with classical machine learning by:
- **Simulating quantum mechanics** to generate enhanced training data
- Using a **classical ANN architecture** that learns from quantum-generated data
- Demonstrating how quantum effects can provide insights and potentially improve ML models

## Quantum Effects Implemented

### 1. **Superposition** 🌊
Quantum states exist in multiple states simultaneously until measured. This creates diverse probability distributions in the training data.

### 2. **Entanglement** 🔗
Creates strong correlations between quantum states. When features are generated from entangled states, they exhibit non-classical correlations.

### 3. **Interference** 〰️
Wave-like behavior where quantum states combine constructively or destructively, creating complex patterns in the data.

### 4. **Measurement** 📊
The collapse of quantum superposition to classical values, introducing quantum randomness into the dataset.

## Installation

```bash
# Clone the repository
git clone https://github.com/svinnapolean/quantum-enhanced-ANN-.git
cd quantum-enhanced-ANN-

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete demonstration:

```bash
python demo.py
```

This will:
1. Demonstrate individual quantum effects
2. Visualize quantum-enhanced data patterns
3. Train an ANN on quantum-generated data
4. Compare quantum-enhanced vs classical random data
5. Generate visualization plots

## Usage Examples

### Generate Quantum-Enhanced Data

```python
from quantum_simulator import QuantumDataGenerator

# Initialize generator
generator = QuantumDataGenerator(seed=42)

# Generate dataset with all quantum effects
X, y = generator.generate_quantum_enhanced_dataset(
    n_samples=1000,
    n_features=10,
    use_superposition=True,
    use_interference=True,
    use_entanglement=True
)
```

### Train Neural Network

```python
from neural_network import NeuralNetwork

# Create network
nn = NeuralNetwork(
    layer_sizes=[10, 16, 8, 1],  # Input -> Hidden layers -> Output
    learning_rate=0.1,
    activation='tanh'
)

# Train on quantum-enhanced data
history = nn.train(X_train, y_train, epochs=100, batch_size=32)

# Evaluate
metrics = nn.evaluate(X_test, y_test)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

### Use Individual Quantum Effects

```python
from quantum_simulator import QuantumSimulator

simulator = QuantumSimulator()

# Create superposition
state = simulator.create_superposition(n_states=4)
print(f"Probabilities: {state.probabilities()}")

# Measure quantum state
measurement = state.measure()
print(f"Measured: {measurement}")

# Create entangled states
state1, state2 = simulator.create_entangled_pair(n_states=2)

# Apply interference
interfered = simulator.interference_pattern(state1, state2, relative_phase=np.pi/4)
```

## Project Structure

```
quantum-enhanced-ANN-/
├── quantum_simulator.py    # Quantum mechanics simulation
├── neural_network.py       # Classical ANN implementation
├── demo.py                 # Complete demonstration script
├── __init__.py            # Package initialization
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Architecture

### Quantum Simulator
- **QuantumState**: Represents quantum states with complex amplitudes
- **QuantumSimulator**: Simulates quantum effects (superposition, entanglement, interference)
- **QuantumDataGenerator**: Generates training data using quantum simulations

### Neural Network
- **NeuralNetwork**: Feedforward ANN with backpropagation
- Configurable architecture (layer sizes, activation functions)
- Training with mini-batch gradient descent
- Comprehensive evaluation metrics

## How It Works

1. **Quantum State Preparation**: Create quantum states using superposition
2. **Apply Quantum Operations**: Introduce entanglement and interference
3. **Measurement**: Collapse quantum states to classical values
4. **Data Generation**: Use measurements to create feature vectors
5. **Classical Learning**: Train ANN on quantum-enhanced data
6. **Evaluation**: Assess performance and compare with classical approaches

## Key Concepts

### Quantum Superposition
A quantum system can exist in multiple states simultaneously. The probability of measuring each state is given by |amplitude|².

### Quantum Entanglement
When quantum systems become entangled, measuring one instantly affects the other, creating correlations stronger than classical physics allows.

### Quantum Interference
Quantum states behave like waves and can interfere constructively (amplifying) or destructively (canceling), affecting measurement probabilities.

### Quantum Measurement
Measurement causes a quantum state to "collapse" from superposition to a definite classical value, introducing fundamental quantum randomness.

## Benefits of Quantum-Enhanced Data

- **Richer feature distributions** from quantum superposition
- **Non-classical correlations** through entanglement
- **Complex patterns** from interference effects
- **Quantum randomness** that differs from classical random noise

## Limitations

This is a **simulation** of quantum effects, not actual quantum computation. It demonstrates:
- How quantum principles can inspire data generation strategies
- The potential of quantum-enhanced machine learning
- Educational insights into quantum computing concepts

For true quantum advantage, actual quantum hardware or advanced quantum algorithms would be needed.

## Dependencies

- NumPy: Numerical computations and array operations
- SciPy: Scientific computing utilities
- Matplotlib: Visualization and plotting

## Future Enhancements

- Integration with actual quantum computing frameworks (Qiskit, Cirq)
- More sophisticated quantum circuits
- Quantum neural network architectures
- Hybrid quantum-classical algorithms
- Comparison with real quantum hardware results

## Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new quantum effects to simulate
- Improve the ANN architecture
- Add more visualization tools
- Enhance documentation

## License

MIT License - Feel free to use and modify for your projects.

## References

- Nielsen & Chuang: "Quantum Computation and Quantum Information"
- Quantum machine learning literature
- Quantum computing frameworks documentation

## Author

Created to demonstrate the fascinating intersection of quantum mechanics and machine learning.

---

**Note**: This is an educational demonstration. For production quantum machine learning applications, consider using established frameworks like Qiskit, TensorFlow Quantum, or PennyLane.
