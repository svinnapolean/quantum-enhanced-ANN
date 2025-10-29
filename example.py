"""
Simple Example: Using Quantum-Enhanced ANN

This is a minimal example showing how to use the quantum-enhanced ANN.
"""

import numpy as np
from quantum_simulator import QuantumDataGenerator
from neural_network import NeuralNetwork


def main():
    """Simple example of training an ANN with quantum-enhanced data."""
    
    print("Quantum-Enhanced ANN - Simple Example")
    print("=" * 50)
    
    # Step 1: Generate quantum-enhanced training data
    print("\n1. Generating quantum-enhanced data...")
    generator = QuantumDataGenerator(seed=42)
    
    X_train, y_train = generator.generate_quantum_enhanced_dataset(
        n_samples=500,      # 500 training samples
        n_features=6,       # 6 features per sample
        use_superposition=True,
        use_interference=True,
        use_entanglement=True
    )
    
    X_test, y_test = generator.generate_quantum_enhanced_dataset(
        n_samples=100,      # 100 test samples
        n_features=6
    )
    
    print(f"   Training data: {X_train.shape}")
    print(f"   Test data: {X_test.shape}")
    print(f"   Class distribution: {np.bincount(y_train)}")
    
    # Step 2: Create neural network
    print("\n2. Creating neural network...")
    nn = NeuralNetwork(
        layer_sizes=[6, 10, 1],    # 6 inputs -> 10 hidden -> 1 output
        learning_rate=0.1,
        activation='tanh',
        seed=42
    )
    print("   Network architecture: [6, 10, 1]")
    
    # Step 3: Train the network
    print("\n3. Training neural network...")
    history = nn.train(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        verbose=False
    )
    
    print(f"   Initial loss: {history['train_loss'][0]:.4f}")
    print(f"   Final loss: {history['train_loss'][-1]:.4f}")
    print(f"   Final accuracy: {history['train_accuracy'][-1]:.4f}")
    
    # Step 4: Evaluate on test data
    print("\n4. Evaluating on test data...")
    metrics = nn.evaluate(X_test, y_test)
    
    print(f"   Test accuracy: {metrics['accuracy']:.4f}")
    print(f"   Test precision: {metrics['precision']:.4f}")
    print(f"   Test recall: {metrics['recall']:.4f}")
    print(f"   Test F1-score: {metrics['f1_score']:.4f}")
    
    # Step 5: Make predictions
    print("\n5. Making predictions on new data...")
    sample_data = X_test[:5]  # Take 5 samples
    predictions = nn.predict(sample_data)
    probabilities = nn.predict_proba(sample_data)
    
    print("\n   Sample predictions:")
    for i in range(5):
        print(f"   Sample {i+1}: Predicted class = {predictions[i]}, "
              f"Probability = {probabilities[i]:.4f}, "
              f"True class = {y_test[i]}")
    
    print("\n" + "=" * 50)
    print("Example complete!")
    print("\nKey takeaways:")
    print("- Quantum effects create unique data patterns")
    print("- Classical ANNs can learn from quantum data")
    print("- The framework is easy to use and extend")
    print("\nFor more details, run: python demo.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
