"""
Quantum-Enhanced ANN Demo

This script demonstrates how quantum mechanics simulation enhances
classical artificial neural networks by generating training data with
quantum effects: superposition, entanglement, interference, and measurement.
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_simulator import QuantumDataGenerator, QuantumSimulator
from neural_network import NeuralNetwork


def demonstrate_quantum_effects():
    """Demonstrate individual quantum effects."""
    print("=" * 70)
    print("DEMONSTRATING QUANTUM EFFECTS")
    print("=" * 70)
    
    simulator = QuantumSimulator()
    
    # 1. Superposition
    print("\n1. SUPERPOSITION - State exists in multiple states simultaneously")
    print("-" * 70)
    superposition_state = simulator.create_superposition(n_states=4)
    print(f"Amplitudes: {superposition_state.amplitudes}")
    print(f"Probabilities: {superposition_state.probabilities()}")
    measurements = [superposition_state.measure() for _ in range(1000)]
    print(f"Measurement distribution (1000 samples): {np.bincount(measurements)}")
    
    # 2. Entanglement
    print("\n2. ENTANGLEMENT - Correlated quantum states")
    print("-" * 70)
    state1, state2 = simulator.create_entangled_pair(n_states=2)
    print(f"Entangled state 1 probabilities: {state1.probabilities()}")
    print(f"Entangled state 2 probabilities: {state2.probabilities()}")
    
    # 3. Interference
    print("\n3. INTERFERENCE - Wave-like behavior combining states")
    print("-" * 70)
    s1 = simulator.create_superposition(4)
    s2 = simulator.create_superposition(4)
    
    # Constructive interference
    constructive = simulator.interference_pattern(s1, s2, relative_phase=0)
    print(f"Constructive interference (phase=0): {constructive.probabilities()}")
    
    # Destructive interference
    destructive = simulator.interference_pattern(s1, s2, relative_phase=np.pi)
    print(f"Destructive interference (phase=π): {destructive.probabilities()}")
    
    # 4. Measurement
    print("\n4. MEASUREMENT - Quantum state collapse to classical value")
    print("-" * 70)
    state = simulator.create_weighted_superposition(np.array([1.0, 2.0, 3.0, 4.0]))
    print(f"Probabilities before measurement: {state.probabilities()}")
    measurement = state.measure()
    print(f"Measured value (collapsed state): {measurement}")
    print()


def train_quantum_enhanced_ann():
    """Train ANN with quantum-enhanced data."""
    print("=" * 70)
    print("TRAINING ANN WITH QUANTUM-ENHANCED DATA")
    print("=" * 70)
    
    # Generate quantum-enhanced dataset
    print("\nGenerating quantum-enhanced training data...")
    generator = QuantumDataGenerator(seed=42)
    
    # Training data with all quantum effects
    X_train, y_train = generator.generate_quantum_enhanced_dataset(
        n_samples=1000,
        n_features=10,
        use_superposition=True,
        use_interference=True,
        use_entanglement=True
    )
    
    # Validation data
    X_val, y_val = generator.generate_quantum_enhanced_dataset(
        n_samples=200,
        n_features=10,
        use_superposition=True,
        use_interference=True,
        use_entanglement=True
    )
    
    # Test data
    X_test, y_test = generator.generate_quantum_enhanced_dataset(
        n_samples=200,
        n_features=10,
        use_superposition=True,
        use_interference=True,
        use_entanglement=True
    )
    
    print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    # Create and train neural network
    print("\nInitializing Neural Network...")
    nn = NeuralNetwork(
        layer_sizes=[10, 16, 8, 1],  # Input -> Hidden1 -> Hidden2 -> Output
        learning_rate=0.1,
        activation='tanh',
        seed=42
    )
    
    print("Training Neural Network on quantum-enhanced data...")
    print("-" * 70)
    history = nn.train(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        verbose=True,
        X_val=X_val,
        y_val=y_val
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    metrics = nn.evaluate(X_test, y_test)
    
    print(f"\nTest Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              0         1")
    print(f"Actual 0  {metrics['confusion_matrix'][0][0]:5d}    {metrics['confusion_matrix'][0][1]:5d}")
    print(f"       1  {metrics['confusion_matrix'][1][0]:5d}    {metrics['confusion_matrix'][1][1]:5d}")
    
    return history, nn, X_test, y_test


def compare_quantum_vs_classical():
    """Compare quantum-enhanced vs classical random data."""
    print("\n" + "=" * 70)
    print("COMPARISON: QUANTUM-ENHANCED vs CLASSICAL RANDOM DATA")
    print("=" * 70)
    
    generator = QuantumDataGenerator(seed=42)
    
    # Quantum-enhanced data
    print("\nTraining with QUANTUM-ENHANCED data...")
    X_q_train, y_q_train = generator.generate_quantum_enhanced_dataset(
        n_samples=800, n_features=8
    )
    X_q_test, y_q_test = generator.generate_quantum_enhanced_dataset(
        n_samples=200, n_features=8
    )
    
    nn_quantum = NeuralNetwork(layer_sizes=[8, 12, 1], learning_rate=0.1, seed=42)
    nn_quantum.train(X_q_train, y_q_train, epochs=50, verbose=False)
    metrics_quantum = nn_quantum.evaluate(X_q_test, y_q_test)
    
    # Classical random data
    print("Training with CLASSICAL RANDOM data...")
    np.random.seed(42)
    X_c_train = np.random.rand(800, 8)
    X_c_test = np.random.rand(200, 8)
    
    # Generate labels using same strategy
    y_c_train = generator._generate_labels(X_c_train)
    y_c_test = generator._generate_labels(X_c_test)
    
    nn_classical = NeuralNetwork(layer_sizes=[8, 12, 1], learning_rate=0.1, seed=42)
    nn_classical.train(X_c_train, y_c_train, epochs=50, verbose=False)
    metrics_classical = nn_classical.evaluate(X_c_test, y_c_test)
    
    print("\n" + "-" * 70)
    print("RESULTS COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<20} {'Quantum-Enhanced':<20} {'Classical Random':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {metrics_quantum['accuracy']:<20.4f} {metrics_classical['accuracy']:<20.4f}")
    print(f"{'Precision':<20} {metrics_quantum['precision']:<20.4f} {metrics_classical['precision']:<20.4f}")
    print(f"{'Recall':<20} {metrics_quantum['recall']:<20.4f} {metrics_classical['recall']:<20.4f}")
    print(f"{'F1 Score':<20} {metrics_quantum['f1_score']:<20.4f} {metrics_classical['f1_score']:<20.4f}")
    print("-" * 70)


def visualize_quantum_data():
    """Visualize quantum-enhanced data patterns."""
    print("\n" + "=" * 70)
    print("VISUALIZING QUANTUM DATA PATTERNS")
    print("=" * 70)
    
    generator = QuantumDataGenerator(seed=42)
    
    # Generate different types of quantum data
    superposition_data = generator.generate_superposition_data(500, 2)
    interference_data = generator.generate_interference_data(500, 2)
    entangled_data = generator.generate_entangled_data(250, 1)  # Creates 2 features
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Superposition data
    axes[0, 0].scatter(superposition_data[:, 0], superposition_data[:, 1], 
                       alpha=0.5, s=20, c='blue')
    axes[0, 0].set_title('Superposition Data', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Interference data
    axes[0, 1].scatter(interference_data[:, 0], interference_data[:, 1], 
                       alpha=0.5, s=20, c='green')
    axes[0, 1].set_title('Interference Data', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Entangled data
    axes[1, 0].scatter(entangled_data[:, 0], entangled_data[:, 1], 
                       alpha=0.5, s=20, c='red')
    axes[1, 0].set_title('Entangled Data (Correlated Features)', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Feature 1')
    axes[1, 0].set_ylabel('Feature 2 (Entangled)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined quantum-enhanced data
    X_combined, y_combined = generator.generate_quantum_enhanced_dataset(
        500, 2, use_superposition=True, use_interference=True, use_entanglement=True
    )
    scatter = axes[1, 1].scatter(X_combined[:, 0], X_combined[:, 1], 
                                 c=y_combined, alpha=0.6, s=20, cmap='coolwarm')
    axes[1, 1].set_title('Combined Quantum-Enhanced Data\n(colored by label)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Feature 1')
    axes[1, 1].set_ylabel('Feature 2')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Class')
    
    plt.tight_layout()
    plt.savefig('quantum_data_visualization.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to 'quantum_data_visualization.png'")
    plt.close()


def plot_training_history(history):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    if history['val_accuracy']:
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining history plot saved to 'training_history.png'")
    plt.close()


def main():
    """Run the complete quantum-enhanced ANN demonstration."""
    print("\n" + "=" * 70)
    print(" " * 15 + "QUANTUM-ENHANCED ARTIFICIAL NEURAL NETWORK")
    print(" " * 20 + "Demonstration and Training")
    print("=" * 70)
    print("\nThis demonstration shows how quantum mechanics principles")
    print("(superposition, entanglement, interference, and measurement)")
    print("can be used to generate enhanced training data for classical ANNs.")
    print("=" * 70)
    
    # Step 1: Demonstrate quantum effects
    demonstrate_quantum_effects()
    
    # Step 2: Visualize quantum data
    visualize_quantum_data()
    
    # Step 3: Train quantum-enhanced ANN
    history, trained_nn, X_test, y_test = train_quantum_enhanced_ann()
    
    # Step 4: Plot training history
    plot_training_history(history)
    
    # Step 5: Compare quantum vs classical
    compare_quantum_vs_classical()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("1. Quantum superposition creates diverse state distributions")
    print("2. Quantum entanglement introduces correlations between features")
    print("3. Quantum interference creates wave-like patterns in data")
    print("4. Measurement collapses quantum states to classical values")
    print("5. Classical ANNs can learn from quantum-enhanced data patterns")
    print("\nGenerated files:")
    print("  - quantum_data_visualization.png")
    print("  - training_history.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
