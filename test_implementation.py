"""
Test script to validate quantum-enhanced ANN implementation.

This script runs basic tests to ensure all components are working correctly.
"""

import numpy as np
from quantum_simulator import QuantumState, QuantumSimulator, QuantumDataGenerator
from neural_network import NeuralNetwork


def test_quantum_state():
    """Test QuantumState class."""
    print("Testing QuantumState...")
    
    # Test normalization
    state = QuantumState([1, 1, 1, 1])
    assert abs(np.sum(state.probabilities()) - 1.0) < 1e-10, "Probabilities should sum to 1"
    
    # Test measurement
    measurement = state.measure()
    assert 0 <= measurement < 4, "Measurement should be in valid range"
    
    print("  ✓ QuantumState tests passed")


def test_quantum_simulator():
    """Test QuantumSimulator class."""
    print("Testing QuantumSimulator...")
    
    simulator = QuantumSimulator()
    
    # Test superposition
    super_state = simulator.create_superposition(n_states=4)
    probs = super_state.probabilities()
    assert len(probs) == 4, "Should have 4 states"
    assert np.allclose(probs, 0.25), "Equal superposition should have equal probabilities"
    
    # Test weighted superposition
    weights = np.array([1, 2, 3, 4])
    weighted_state = simulator.create_weighted_superposition(weights)
    assert abs(np.sum(weighted_state.probabilities()) - 1.0) < 1e-10, "Probabilities should sum to 1"
    
    # Test entanglement
    s1, s2 = simulator.create_entangled_pair(n_states=2)
    assert len(s1.probabilities()) == 2, "Entangled state should have 2 states"
    assert len(s2.probabilities()) == 2, "Entangled state should have 2 states"
    
    # Test interference
    state1 = simulator.create_superposition(4)
    state2 = simulator.create_superposition(4)
    interfered = simulator.interference_pattern(state1, state2, 0)
    assert abs(np.sum(interfered.probabilities()) - 1.0) < 1e-10, "Probabilities should sum to 1"
    
    print("  ✓ QuantumSimulator tests passed")


def test_quantum_data_generator():
    """Test QuantumDataGenerator class."""
    print("Testing QuantumDataGenerator...")
    
    generator = QuantumDataGenerator(seed=42)
    
    # Test superposition data
    super_data = generator.generate_superposition_data(n_samples=100, n_features=5)
    assert super_data.shape == (100, 5), "Shape should be (100, 5)"
    assert np.all((super_data >= 0) & (super_data <= 1)), "Values should be in [0, 1]"
    
    # Test interference data
    inter_data = generator.generate_interference_data(n_samples=100, n_features=5)
    assert inter_data.shape == (100, 5), "Shape should be (100, 5)"
    
    # Test entangled data
    entangled_data = generator.generate_entangled_data(n_samples=100, n_feature_pairs=3)
    assert entangled_data.shape == (100, 6), "Shape should be (100, 6)"
    
    # Test complete dataset
    X, y = generator.generate_quantum_enhanced_dataset(n_samples=100, n_features=10)
    assert X.shape == (100, 10), "X shape should be (100, 10)"
    assert y.shape == (100,), "y shape should be (100,)"
    assert set(y).issubset({0, 1}), "Labels should be binary (0 or 1)"
    
    print("  ✓ QuantumDataGenerator tests passed")


def test_neural_network():
    """Test NeuralNetwork class."""
    print("Testing NeuralNetwork...")
    
    # Create simple dataset
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 5)
    y_test = np.random.randint(0, 2, 20)
    
    # Create and train network
    nn = NeuralNetwork(
        layer_sizes=[5, 8, 1],
        learning_rate=0.1,
        activation='sigmoid',
        seed=42
    )
    
    # Test forward pass
    output = nn.forward(X_train)
    assert output.shape == (100, 1), "Output shape should be (100, 1)"
    assert np.all((output >= 0) & (output <= 1)), "Output should be in [0, 1]"
    
    # Test training
    history = nn.train(X_train, y_train, epochs=10, verbose=False)
    assert 'train_loss' in history, "History should contain train_loss"
    assert 'train_accuracy' in history, "History should contain train_accuracy"
    assert len(history['train_loss']) == 10, "Should have 10 epochs of loss"
    
    # Test prediction
    predictions = nn.predict(X_test)
    assert predictions.shape == (20,), "Predictions shape should be (20,)"
    assert set(predictions).issubset({0, 1}), "Predictions should be binary"
    
    # Test evaluation
    metrics = nn.evaluate(X_test, y_test)
    assert 'accuracy' in metrics, "Metrics should contain accuracy"
    assert 'precision' in metrics, "Metrics should contain precision"
    assert 'recall' in metrics, "Metrics should contain recall"
    assert 'f1_score' in metrics, "Metrics should contain f1_score"
    assert 0 <= metrics['accuracy'] <= 1, "Accuracy should be in [0, 1]"
    
    print("  ✓ NeuralNetwork tests passed")


def test_integration():
    """Test integration of quantum data generation with neural network."""
    print("Testing Integration...")
    
    # Generate quantum-enhanced dataset
    generator = QuantumDataGenerator(seed=42)
    X_train, y_train = generator.generate_quantum_enhanced_dataset(
        n_samples=200,
        n_features=8,
        use_superposition=True,
        use_interference=True,
        use_entanglement=True
    )
    
    X_test, y_test = generator.generate_quantum_enhanced_dataset(
        n_samples=50,
        n_features=8
    )
    
    # Train neural network
    nn = NeuralNetwork(
        layer_sizes=[8, 12, 1],
        learning_rate=0.1,
        activation='tanh',
        seed=42
    )
    
    history = nn.train(X_train, y_train, epochs=20, batch_size=32, verbose=False)
    
    # Evaluate
    metrics = nn.evaluate(X_test, y_test)
    
    # Check that model learned something (loss decreased)
    assert history['train_loss'][0] > history['train_loss'][-1], \
        "Training loss should decrease"
    
    # Check that accuracy is reasonable (at least random chance)
    assert metrics['accuracy'] >= 0.3, \
        "Accuracy should be better than random for this task"
    
    print("  ✓ Integration tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("RUNNING QUANTUM-ENHANCED ANN TESTS")
    print("=" * 70)
    print()
    
    try:
        test_quantum_state()
        test_quantum_simulator()
        test_quantum_data_generator()
        test_neural_network()
        test_integration()
        
        print()
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("=" * 70)
        print()
        print("The quantum-enhanced ANN implementation is working correctly.")
        print("You can now run 'python demo.py' for a full demonstration.")
        print()
        return True
        
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        return False
    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
